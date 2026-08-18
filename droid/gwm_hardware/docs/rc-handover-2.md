# Handover #2 to the RC machine — network answer, and two asks

Your Step-6 report landed. libfranka **0.9.2** is exactly right for a Panda on
system 4.2.2, and the gripper work below is a genuine catch. Status from this
side first, then the answer you asked for.

## Verified from zhiwei just now

Reading through tiptop's own client factory (`get_robot_client()` →
`BambooFrankaClient`), **no motion commands sent**:

```
config  : type=panda_robotiq host=192.168.68.132 ports=5555/5559
qpos    : [-0.0135, -0.7743, 0.0083, -2.3673, -0.0135, 1.5698, 0.7797]
ee xyz  : [0.3093, -0.0024, 0.5816]
gripper_state     : 0.14      <- your stroke patch is visible from here
RPC     : median 18.1 ms, p95 142.3 ms, max 144.6 ms (client timeout 5000 ms)
```

Both ports are reachable. Earlier today 5559 was closed; it opened once you got
the gripper activated, so nothing is wrong there.

## 1. Which network — answer: `192.168.68.132` for now

Measured from zhiwei (`192.168.68.133/22`, WiFi):

| Candidate | Result |
|---|---|
| **192.168.68.132** (your WiFi) | ✅ reachable; 30 pings 0 % loss, avg 5.98 ms, **max 67.5 ms, mdev 11.8 ms** |
| 192.168.0.16 (your USB wired) | ❌ **unreachable** — different subnet, routed at `192.168.68.1` and dropped |
| 100.80.197.100 (Tailscale) | ✅ reachable but avg 29.3 ms, max 81.9 ms, mdev 37 ms |

`192.168.0.16` is not an option today: zhiwei has no route onto `192.168.0.x`.
Tailscale is pointless for two hosts on the same WiFi — it wraps WireGuard
around a link that is already direct and makes the jitter 150x worse.

**So: `robot.host: 192.168.68.132`.** Already written into
`gwm_hardware/config/tiptop.yml` on this side.

### On your "WiFi 不建议跑轨迹" note

Good instinct, wrong failure mode here — worth being precise because it changes
what we should worry about.

This link carries **no real-time traffic**. `execute_cutamp_plan` hands Bamboo
an entire trajectory in one ZMQ call (`execute_joint_impedance_path` with the
full `positions`/`velocities`/`durations` arrays); the 1 kHz joint-impedance
loop runs entirely inside your machine. A WiFi hiccup therefore delays the
*upload*, or fails the RPC cleanly **before** motion starts — it cannot make
the motion itself jerky. Measured p95 of 142 ms against a 5000 ms timeout
leaves a lot of headroom.

The one real risk is a drop *during* execution: the client blocks waiting for
the completion reply, times out, and raises `ExecutionFailure` while your side
happily finishes the motion. Not dangerous, but it desynchronises the two
machines mid-episode, and it will happen eventually on WiFi.

**So wired is still worth doing, just not urgent.** The cheap path: zhiwei's
`enp6s0` is now free (the robot cable moved to you) and has no carrier. If you
can put a cable from that port onto whatever switch your USB NIC's
`192.168.0.x` hangs off, both machines get a wired path and we re-point
`robot.host` at `192.168.0.16`. Tell me if that switch is reachable from
zhiwei's bench.

## 2. Please tighten `--listen_ip` — bind to the WiFi address

Agreed, `"*"` is too open: anything on the WiFi can currently drive the arm.

```bash
bash RunBambooController stop
bash RunBambooController --listen_ip 192.168.68.132
```

Then tell me, and I will re-run the read-only link check to confirm zhiwei can
still reach both ports. Note this pins us to WiFi — if we later move to the
wired LAN, it needs re-binding to `192.168.0.16`.

## 3. Your four Bamboo patches — yes, commit them, and please send me the diff

These are load-bearing for this rig, not incidental:

- the 3 s → 15 s activation poll (2F-140 self-calibration takes ~3.6 s)
- `stroke` 0.085 → 0.14 in the driver, the gripper server, and `client.py`

Two requests:

1. **Commit them to a local branch** — yes please. A `git checkout` or a
   re-clone silently reverts the gripper to a 2F-85 stroke, and the symptom
   (wrong width/position scaling) is quiet rather than loud.
2. **Send me `git diff` output** so I can version it in this repo at
   `droid/gwm_hardware/patches/bamboo-2f140.patch`. Your bamboo clone is not
   part of the gwm-wiser tree, so today the patch exists on exactly one disk.
   If that machine is ever reimaged we lose it and get to rediscover the 3.6 s
   activation timeout the hard way.

Cross-check while you are there: your `gripper.py` read **0.1400 m** fully
open, and the URDF model I generated on this side measures **136.0 mm** between
the two finger-pad *link origins*. Those are consistent (the pad bodies sit
inside their origins), but if you have a caliper measurement of the actual
inner-face-to-inner-face opening it would be worth having — it feeds the grasp
gate's thresholds later.

## 4. What changed on this side

`tiptop-run` on this rig now plans a **2F-140**, not the 2F-85 cuTAMP ships.
That mattered: the 2F-140's TCP sits **62 mm** further from the flange, so the
stock model would have driven the real fingers 62 mm too deep on every grasp.
Generated model + cuRobo config + grasp-filter spheres live in
`droid/gwm_hardware/`, verified 9/9 (TCP 212.0 mm, gripper 136 → 7 mm through
the mimic chain, IK 0.00 mm, motion planning at 1.94x the 2F-85's cost).

Nothing on this side commands the robot yet.

## 5. What I need from you next

1. Confirm the `--listen_ip` tightening is done (and to which address).
2. `git diff` of the Bamboo patches.
3. Whether zhiwei's bench can reach the `192.168.0.x` switch.
4. **Confirm the Franka Desk end-effector payload describes the 2F-140** —
   mass, centre of mass, inertia tensor, TCP. Your `joint_trajectory.py` run
   passing with no reflex is encouraging, but that was a free-space move; a
   wrong payload shows up as reflex stops under load, i.e. exactly when the
   gripper is holding something.

Once 1–2 are in, the next hardware step is on this side: workspace obstacle
modelling, then `go-to-capture` — the first commanded motion from zhiwei. I
will ask before anything moves.
