"""Plan JSON -> per-control-step actions, exactly as the DROID-sim harness executes them.

A numpy-only port of `TiptopWebsocketClient._step_plan` /
`_subsample_trajectory` (droid-sim-evals/src/sim_evals/inference/tiptop_websocket.py)
so the same timeline can be produced inside the Isaac process (replay) and
offline (state/action sequences for the world model) without a websocket.

Harness facts reproduced here:
  * control runs at 15 Hz; cuRobo waypoints are 50 Hz -> every 3rd waypoint is
    sent, the final waypoint always included;
  * a `gripper` step holds the current joint pose for 20 control steps
    (1.333 s) with the gripper command set to its new value;
  * one action per control step: [q1..q7, gripper], gripper 1.0 = close, 0.0 = open;
  * the harness stops as soon as `plan_done` turns True, i.e. the last waypoint
    is not stepped by the loop itself -- it is applied during the 30-step hold
    that follows (batch_eval_v2.py:226-239).
"""

import json

import numpy as np

SIM_CONTROL_HZ = 15.0
CUROBO_INTERP_HZ = 50.0
WAYPOINT_STRIDE = max(1, int(round(CUROBO_INTERP_HZ / SIM_CONTROL_HZ)))  # 3
GRIPPER_ACTION_STEPS = 20
HOLD_STEPS_AFTER_PLAN = 30


def load_plan(path):
    with open(path) as f:
        plan = json.load(f)
    steps = []
    for s in plan["steps"]:
        if s["type"] == "metadata":
            continue
        s = dict(s)
        if s["type"] == "trajectory":
            s["positions"] = np.asarray(s["positions"], dtype=np.float32)
        steps.append(s)
    return {"q_init": np.asarray(plan["q_init"], dtype=np.float32), "steps": steps, "version": plan.get("version")}


def subsample_trajectory(trajectory, stride=WAYPOINT_STRIDE):
    if stride <= 1 or len(trajectory) == 0:
        return trajectory
    indices = np.arange(0, len(trajectory), stride)
    if indices[-1] != len(trajectory) - 1:
        indices = np.append(indices, len(trajectory) - 1)
    return trajectory[indices]


class PlanStepper:
    """Call `step(joint_position, gripper_position)` once per control step.

    Returns the 8-vector action for that step and records `plan_done` with the
    harness's semantics (True once the action just returned was the last one
    the plan produces). `label` is the plan step the action came from.
    """

    def __init__(self, plan, gripper_action_steps=GRIPPER_ACTION_STEPS, stride=WAYPOINT_STRIDE):
        self._plan = plan["steps"]
        self._gripper_action_steps = gripper_action_steps
        self._stride = stride
        self._current_plan_step = 0
        self._current_trajectory = None
        self._current_waypoint_idx = 0
        self._gripper_action_pending = None
        self._gripper_action_steps_remaining = 0
        self._last_gripper_state = 0.0
        self.label = None
        self.phase = None  # "trajectory" | "gripper" | "hold"

    @property
    def plan_done(self):
        if self._gripper_action_pending is not None:
            return False
        if self._current_trajectory is not None and self._current_waypoint_idx < len(self._current_trajectory):
            return False
        return self._current_plan_step >= len(self._plan)

    @property
    def last_gripper_state(self):
        return self._last_gripper_state

    def step(self, joint_position, gripper_position=None):
        joint_position = np.asarray(joint_position, dtype=np.float32).reshape(-1)
        if self._gripper_action_pending is not None:
            if self._gripper_action_steps_remaining > 0:
                self._gripper_action_steps_remaining -= 1
                gripper_val = 1.0 if self._gripper_action_pending == "close" else 0.0
                self._last_gripper_state = gripper_val
                self.phase = "gripper"
                return np.concatenate([joint_position, [gripper_val]]).astype(np.float32)
            self._last_gripper_state = 1.0 if self._gripper_action_pending == "close" else 0.0
            self._gripper_action_pending = None
            self._current_plan_step += 1

        if self._current_trajectory is None or self._current_waypoint_idx >= len(self._current_trajectory):
            if self._current_plan_step >= len(self._plan):
                # plan completed: hold
                g = self._last_gripper_state if gripper_position is None else float(np.asarray(gripper_position).reshape(-1)[0])
                self.phase = "hold"
                return np.concatenate([joint_position, [g]]).astype(np.float32)
            step = self._plan[self._current_plan_step]
            self.label = step.get("label")
            if step["type"] == "gripper":
                self._gripper_action_pending = step["action"]
                self._gripper_action_steps_remaining = self._gripper_action_steps
                gripper_val = 1.0 if self._gripper_action_pending == "close" else 0.0
                self._last_gripper_state = gripper_val
                self.phase = "gripper"
                return np.concatenate([joint_position, [gripper_val]]).astype(np.float32)
            self._current_trajectory = subsample_trajectory(step["positions"], self._stride)
            self._current_waypoint_idx = 0
            self._current_plan_step += 1

        waypoint = self._current_trajectory[self._current_waypoint_idx]
        self._current_waypoint_idx += 1
        self.phase = "trajectory"
        if waypoint.shape[0] == 7:
            return np.concatenate([waypoint, [self._last_gripper_state]]).astype(np.float32)
        return np.asarray(waypoint, dtype=np.float32)


def unroll_plan(plan, hold_steps=HOLD_STEPS_AFTER_PLAN):
    """Offline unroll: the action the harness sends at every control step.

    Uses the last commanded joint pose wherever the harness would read the
    measured one (gripper holds), which is the idealised no-tracking-error
    timeline. Returns dict with
      actions  [N, 8]  per-step commands (the loop's steps, then `hold_steps`
                       repeats of the final action, as in batch_eval_v2)
      t        [N]     time of each step from plan start (s), k / 15
      phase    [N]     'trajectory' | 'gripper' | 'hold'
      label    [N]     plan step label
      n_loop   int     steps the harness loop itself executes before plan_done
      close_t  float   time of the first close command (None if never)
    """
    stepper = PlanStepper(plan)
    q = plan["q_init"].copy()
    actions, phases, labels = [], [], []
    last_action = None
    while True:
        a = stepper.step(q)
        if stepper.plan_done:
            last_action = a
            break
        actions.append(a)
        phases.append(stepper.phase)
        labels.append(stepper.label)
        q = a[:7]
    n_loop = len(actions)
    for _ in range(hold_steps):
        actions.append(last_action)
        phases.append("hold")
        labels.append("hold")
    actions = np.stack(actions).astype(np.float32)
    t = np.arange(len(actions)) / SIM_CONTROL_HZ
    close_idx = np.where(actions[:, 7] > 0.5)[0]
    close_t = float(t[close_idx[0]]) if len(close_idx) else None
    return {
        "actions": actions,
        "t": t,
        "phase": np.asarray(phases),
        "label": np.asarray(labels),
        "n_loop": n_loop,
        "close_t": close_t,
    }
