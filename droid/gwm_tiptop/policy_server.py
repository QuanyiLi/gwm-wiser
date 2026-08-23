"""Websocket plan server for selection-policy evaluation runs (no planner, no GPU).

Speaks the tiptop_websocket_server protocol (metadata on connect, msgpack
request in, JSON plan out) but serves precomputed serialize_plan JSONs from a
proposals dir instead of planning, so batch_eval_v2 / grasp_eval run unchanged
against it (--ws-port). Policies:

- ``random``  — uniform over the proposals_index.json candidates (the
  instruction-blind baseline).
- ``fixed``   — always serve --plan-file (the GWM-selected plan; selection is
  done offline by score_client against gwm-server).
- ``proxy``   — forward the request to the real tiptop-server and serve ITS
  plan, optionally truncated at ``--truncate-after`` (the tiptop baseline for
  the place eval). In the place eval the held block is welded to the gripper
  for the whole episode, so tiptop's native tail (open gripper + GoToInitial)
  would carry the block back home. Truncating at the last Place trajectory
  step ends the episode exactly where the GWM place candidates end — block
  inside the chosen bin, still held — so both policies produce the same
  episode shape. Nothing else about tiptop is changed: it still does its own
  per-trial perception + Gemini grounding + cuTAMP planning.

Every request logs the served candidate to --log-jsonl and checks the
request's q_init against the plan's stored q_init (catches a sim reset state
that drifted from the capture the proposals were planned on).

Run in the droid-sim-evals venv (only needs websockets + msgpack_numpy):

    ../droid-sim-evals/.venv/bin/python policy_server.py \
        --proposals-dir ../gwm_integrate_doc/proposals/scene1 \
        --select random --seed 0 --port 8766 --log-jsonl served.jsonl
    ../droid-sim-evals/.venv/bin/python policy_server.py \
        --select proxy --upstream-port 8765 --truncate-after Place --port 8770
"""

import argparse
import asyncio
import json
import logging
import random
import time
from pathlib import Path

import msgpack_numpy
import numpy as np
import websockets.asyncio.client as ws_client
import websockets.asyncio.server as ws_server

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
_log = logging.getLogger("policy_server")


def truncate_plan(plan: dict, prefix: str) -> dict:
    """Drop everything after the LAST trajectory step whose label starts with `prefix`.

    tiptop plans end `... Place(...) | gripper open | GoToInitial(q0)`; keeping
    up to the final Place trajectory leaves the object at the placement pose,
    still held — the episode shape the GWM place candidates were built for.
    """
    steps = plan.get("steps", [])
    keep = max(
        (i for i, s in enumerate(steps)
         if s.get("type") == "trajectory" and str(s.get("label", "")).startswith(prefix)),
        default=None,
    )
    if keep is None:
        _log.warning(f"truncate_after={prefix!r}: no matching trajectory step in "
                     f"{[s.get('label') for s in steps]} — serving the full plan")
        return plan
    return {**plan, "steps": steps[: keep + 1]}


class PolicyServer:
    def __init__(self, args):
        self.args = args
        self.n_requests = 0
        if args.select == "fixed":
            if args.plan_file is None:
                raise SystemExit("--select fixed requires --plan-file")
            self.fixed_plan = json.loads(Path(args.plan_file).read_text())
            self.fixed_name = Path(args.plan_file).name
            self.candidates = None
        elif args.select == "proxy":
            self.candidates = None
            self.upstream = f"ws://{args.upstream_host}:{args.upstream_port}"
            _log.info(f"proxying to {self.upstream}, truncate_after={args.truncate_after!r}")
        else:
            index = json.loads((args.proposals_dir / "proposals_index.json").read_text())
            self.candidates = [
                (e["file"], e["target"], json.loads((args.proposals_dir / e["file"]).read_text()))
                for e in index["proposals"]
            ]
            self.rng = random.Random(args.seed)
            _log.info(f"Loaded {len(self.candidates)} candidates from {args.proposals_dir}")
        self.metadata = {"server": "gwm_tiptop_policy", "select": args.select, "version": "0.1.0"}

    def pick(self, obs: dict) -> tuple[str, str, dict]:
        if self.args.select == "fixed":
            return self.fixed_name, "fixed", self.fixed_plan
        name, target, plan = self.rng.choice(self.candidates)
        return name, target, plan

    async def resolve(self, raw: bytes, obs: dict) -> tuple[str, str, dict | None, str | None]:
        """(name, target, plan, error). Only the proxy policy can fail to produce a plan."""
        if self.args.select != "proxy":
            return (*self.pick(obs), None)
        async with ws_client.connect(self.upstream, compression=None, max_size=None) as up:
            await up.recv()  # server metadata (msgpack), discarded
            await up.send(raw)
            reply = json.loads(await up.recv())
        if not reply.get("success"):
            return "upstream", "tiptop", None, str(reply.get("error"))[:200]
        plan = reply["plan"]
        labels = [s.get("label") for s in plan.get("steps", [])]
        if self.args.truncate_after:
            plan = truncate_plan(plan, self.args.truncate_after)
        return f"tiptop[{len(plan['steps'])}/{len(labels)}]", "tiptop", plan, None

    async def handler(self, websocket) -> None:
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self.metadata))
        while True:
            try:
                raw = await websocket.recv()
            except Exception:
                return  # client closed (per-episode reconnect churn is normal)
            t0 = time.monotonic()
            obs = msgpack_numpy.unpackb(raw)
            req = self.n_requests
            self.n_requests += 1
            try:
                name, target, plan, error = await self.resolve(raw, obs)
            except Exception as e:  # upstream unreachable / protocol error
                name, target, plan, error = "upstream", "tiptop", None, f"{type(e).__name__}: {e}"[:200]

            q_diff = float("nan")
            if plan is not None:
                q_req = np.asarray(obs.get("q_init"), dtype=np.float64).ravel()
                q_plan = np.asarray(plan.get("q_init"), dtype=np.float64).ravel()
                q_diff = float(np.abs(q_req - q_plan).max()) if q_req.size == q_plan.size else float("nan")
                if not q_diff < 1e-3:
                    _log.warning(f"req {req}: q_init drift {q_diff:.4f} rad vs plan capture — scene may not match proposals")

            record = {"req": req, "task": str(obs.get("task")), "file": name,
                      "target": target, "q_init_maxdiff": q_diff, "error": error}
            if plan is not None:
                record["steps"] = [s.get("label") for s in plan.get("steps", [])]
            _log.info(f"req {req}: task={record['task']!r} -> {name} (target={target}, qdiff={q_diff:.2e}"
                      + (f", ERROR {error}" if error else "") + ")")
            if self.args.log_jsonl:
                with open(self.args.log_jsonl, "a") as f:
                    f.write(json.dumps(record) + "\n")

            result = {
                "success": plan is not None,
                "plan": plan,
                "error": error,
                "save_dir": "",
                "server_timing": {"infer_ms": (time.monotonic() - t0) * 1000},
            }
            await websocket.send(json.dumps(result))

    async def run(self) -> None:
        _log.info(f"PolicyServer ({self.args.select}) on ws://0.0.0.0:{self.args.port}")
        async with ws_server.serve(self.handler, "0.0.0.0", self.args.port,
                                   compression=None, max_size=None) as server:
            await server.serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposals-dir", type=Path)
    ap.add_argument("--select", choices=["random", "fixed", "proxy"], required=True)
    ap.add_argument("--plan-file", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--log-jsonl", type=Path)
    ap.add_argument("--upstream-host", default="localhost")
    ap.add_argument("--upstream-port", type=int, default=8765)
    ap.add_argument("--truncate-after", default=None, metavar="LABEL_PREFIX",
                    help="proxy only: keep steps up to the last trajectory step "
                         "whose label starts with this prefix (e.g. Place)")
    args = ap.parse_args()
    if args.select == "random" and args.proposals_dir is None:
        raise SystemExit("--select random requires --proposals-dir")
    asyncio.run(PolicyServer(args).run())


if __name__ == "__main__":
    main()
