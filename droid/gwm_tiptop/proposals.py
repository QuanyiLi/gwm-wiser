"""run_proposals: K executable pick trajectories via cuTAMP as an unmodified library (G-4/G-5/G-6).

For every anonymous cluster we plan toward Holding(object_i), rank the
satisfying particles, pick a diverse subset via confidence-weighted SE(3)
farthest-point sampling over grasp poses, and motion-refine each selected
particle — collecting every success instead of stopping at the first.
"""

import logging

import numpy as np
import torch
from cutamp.algorithm import (
    ParticleOptimizer,
    get_ranked_satisfying_particles,
    sample_plan_skeleton,
    setup_cutamp,
)
from cutamp.config import TAMPConfiguration
from cutamp.constraint_checker import ConstraintChecker
from cutamp.cost_reduction import CostReducer
from cutamp.envs import TAMPEnvironment
from cutamp.utils.common import get_world_cfg
from cutamp.motion_solver import MotionPlanningError, solve_curobo
from cutamp.particle_initialization import ParticleInitializer
from cutamp.scripts.utils import default_constraint_to_mult, default_constraint_to_tol
from cutamp.tamp_domain import Holding, PlaceNear, all_tamp_operators
from cutamp.task_planning import task_plan_generator
from cutamp.task_planning.constraints import StablePlacement

_log = logging.getLogger(__name__)


def _grasp_pose_key(particles: dict) -> str | None:
    """Find the particle key holding grasp poses (N, 4, 4)."""
    for k, v in particles.items():
        if k.startswith("grasp") and not k.endswith("_confidences") and torch.is_tensor(v) and v.ndim == 3:
            return k
    return None


def se3_fps_indices(poses: np.ndarray, confidences: np.ndarray | None, k: int, rot_weight: float = 0.1) -> list[int]:
    """Confidence-weighted farthest-point sampling over SE(3) grasp poses.

    Distance = translation distance + rot_weight * geodesic rotation distance.
    Selection starts at the highest-confidence pose; each next pick maximizes
    (min distance to selected) * confidence — diversity first, confidence as a
    tie-breaker (M2T2 confidence is only used within an object, per G-6).
    """
    n = len(poses)
    if n <= k:
        return list(range(n))
    conf = np.ones(n) if confidences is None else np.clip(np.asarray(confidences), 1e-3, None)
    t = poses[:, :3, 3]
    R = poses[:, :3, :3]
    selected = [int(np.argmax(conf))]
    d_min = np.full(n, np.inf)
    while len(selected) < k:
        last = selected[-1]
        dt = np.linalg.norm(t - t[last], axis=1)
        tr = np.clip((np.einsum("nij,ij->n", R @ R[last].T, np.eye(3)) - 1) / 2, -1, 1)
        dr = np.arccos(tr)
        d_min = np.minimum(d_min, dt + rot_weight * dr)
        d_min[selected] = -np.inf
        selected.append(int(np.argmax(d_min * conf)))
    return selected


def build_scorers(all_surfaces) -> tuple[CostReducer, ConstraintChecker]:
    """Same tolerance loosening as tiptop.planning.run_planning."""
    constraint_to_tol = default_constraint_to_tol.copy()
    constraint_to_mult = default_constraint_to_mult.copy()
    for surface in all_surfaces:
        constraint_to_tol[StablePlacement.type][f"{surface.name}_in_xy"] = 1e-2
        constraint_to_tol[StablePlacement.type][f"{surface.name}_support"] = 1e-2
        constraint_to_mult[StablePlacement.type][f"{surface.name}_support"] = 1.0
    return CostReducer(constraint_to_mult), ConstraintChecker(constraint_to_tol)


def run_proposals(
    env: TAMPEnvironment,
    config: TAMPConfiguration,
    q_init: np.ndarray,
    ik_solver,
    grasps: dict,
    motion_gen,
    all_surfaces: list,
    k_total: int = 16,
    max_refine_per_candidate_slack: int = 2,
) -> list[dict]:
    """Plan Holding(object_i) for every movable and return all refined trajectories.

    Returns a list of {"target", "grasp_confidence", "steps"} dicts, where
    "steps" is a cuTAMP plan (solve_curobo output) ready for
    tiptop.planning.serialize_plan.
    """
    cost_reducer, constraint_checker = build_scorers(all_surfaces)
    exp_logger, visualizer, timer, world = setup_cutamp(env, config, q_init, ik_solver=ik_solver)
    _log.info(
        f"world movables: {[m.name for m in world.movables]}, statics: {[s.name for s in world.statics]}, "
        f"sphere keys: {list(world._obj_to_spheres.keys())}"
    )
    particle_initializer = ParticleInitializer(world, config, grasps)
    particle_optimizer = ParticleOptimizer(config, cost_reducer, constraint_checker)
    operators = [op for op in all_tamp_operators if op is not PlaceNear]

    movable_labels = [m.name for m in world.movables]
    if not movable_labels:
        raise ValueError("No movables in environment")
    # k_total is a scene-independent budget; split it over however many clusters
    # perception found (floor + remainder), so exactly k_total come out no matter
    # the object count. Per-object shortfalls are not redistributed.
    n_obj = len(movable_labels)
    quotas = [k_total // n_obj + (1 if i < k_total % n_obj else 0) for i in range(n_obj)]
    _log.info(f"Proposing for {n_obj} objects, quotas {quotas} (budget {k_total})")

    # Cache initial poses before touching cuRobo (mirrors run_cutamp's ordering),
    # then update the motion-gen world once.
    obj_to_initial_pose = {obj.name: world.get_object_pose(obj) for obj in world.movables}
    if motion_gen is not None:
        # cuTAMP v0.0.6 landmine: get_world_cfg(include_movables=True) does
        # `obstacles = env.movables; obstacles += env.statics`, extending
        # env.movables IN PLACE (statics leak into the movables list and the
        # cost function later requests collision spheres for the table).
        # Hand it a throwaway env snapshot instead.
        env_snapshot = TAMPEnvironment(
            name=env.name,
            movables=list(env.movables),
            statics=list(env.statics),
            type_to_objects=env.type_to_objects,
            goal_state=env.goal_state,
        )
        motion_gen.update_world(get_world_cfg(env_snapshot, include_movables=True))

    # ParticleOptimizer reads this timer for its max_loop_dur budget
    timer.start("start_optimization")

    proposals: list[dict] = []
    for label, k_per in zip(movable_labels, quotas):
        goal = frozenset({Holding.ground(label)})
        plan_gen = task_plan_generator(
            world.initial_state, goal, operators=operators, explored_state_check=config.explored_state_check
        )
        try:
            plan_info, _ = sample_plan_skeleton(
                plan_gen, world, config, timer, 0, constraint_checker, cost_reducer, particle_initializer
            )
        except StopIteration:
            plan_info = None
        if plan_info is None:
            _log.warning(f"{label}: no plan skeleton / particle init failed, skipping")
            continue
        _log.info(f"{label}: skeleton = {[str(op) for op in plan_info['plan_skeleton']]}")

        has_satisfying, metrics, _ = particle_optimizer(plan_info, timer, visualizer)
        # ParticleOptimizer leaks this timer on its early-satisfied exit path;
        # a second call would then raise "Timer already started".
        if timer.has_timer("optimization_step"):
            timer.stop("optimization_step")
        if not has_satisfying:
            _log.warning(f"{label}: no satisfying particles after optimization, skipping")
            continue

        ranked = get_ranked_satisfying_particles(plan_info, config, constraint_checker, cost_reducer)
        num_sat = next(iter(ranked.values())).shape[0]

        gkey = _grasp_pose_key(ranked)
        if gkey is not None:
            poses = ranked[gkey].detach().cpu().numpy()
            confs = None
            ckey = f"{gkey}_confidences"
            if ckey in ranked and ranked[ckey] is not None:
                confs = ranked[ckey].detach().cpu().numpy()
            order = se3_fps_indices(poses, confs, min(k_per * max_refine_per_candidate_slack, num_sat))
        else:
            _log.warning(f"{label}: no grasp-pose key in particles; falling back to rank order")
            order = list(range(num_sat))

        successes = 0
        for j, idx in enumerate(order):
            if successes >= k_per:
                break
            particle = {k: v[idx] for k, v in ranked.items()}
            try:
                steps = solve_curobo(
                    plan_info,
                    particle,
                    world,
                    config,
                    timer,
                    visualizer,
                    obj_to_initial_pose=obj_to_initial_pose,
                    timeline=f"curobo_{label}_{j}",
                    motion_gen=motion_gen,
                )
            except MotionPlanningError as e:
                _log.info(f"{label}: refine {j} failed ({e})")
                continue
            conf = None
            if gkey is not None and f"{gkey}_confidences" in ranked and ranked[f"{gkey}_confidences"] is not None:
                conf = float(ranked[f"{gkey}_confidences"][idx].item())
            proposals.append({"target": label, "grasp_confidence": conf, "steps": steps})
            successes += 1
        _log.info(f"{label}: {successes} executable candidates ({num_sat} satisfying particles)")

    _log.info(f"run_proposals: {len(proposals)} total candidates across {len(movable_labels)} objects")
    return proposals
