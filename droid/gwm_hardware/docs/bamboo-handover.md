# Handover: set up and verify the Bamboo controller on this machine

You are on the **control workstation** of a two-machine robot rig. Another
Claude Code session is running on the GPU workstation (`zhiwei`) and has
already brought up the perception/planning stack there. Your job is the
low-level robot controller. Read the whole brief before running anything —
there are two version constraints that are easy to get wrong and expensive to
get wrong.

## The rig

| | |
|---|---|
| Arm | **Franka Emika Panda** (NOT an FR3) |
| Robot system version | **4.2.2** (read off Franka Desk) |
| Robot IP | `172.16.0.2` (Desk at `https://172.16.0.2/desk/`) |
| Gripper | **Robotiq 2F-140** (NOT the 2F-85 that DROID/TiPToP assume) |
| Your machine | must have a **PREEMPT_RT kernel** and a direct wired link to the Franka controller |
| GPU workstation | `zhiwei`, runs TiPToP + cuRobo/cuTAMP + M2T2 + FoundationStereo |

## What you are building and why

`zhiwei` does perception, motion planning and model inference. It does **not**
talk to the robot directly. It sends finished joint trajectories
(`positions`, `velocities`, `dt` per waypoint) and gripper open/close commands
over ZMQ to **Bamboo** on your machine; Bamboo runs the 1 kHz joint-impedance
control loop against the Franka via libfranka. That 1 kHz hard-real-time loop
is the whole reason this machine needs the RT kernel.

```
zhiwei (RTX 5090)                        this machine (RT kernel)
  cameras -> depth -> point cloud
  M2T2 grasps + cuTAMP/cuRobo plan
  GWM scores and picks a trajectory
        |                                     Bamboo
        +-- joint trajectory ---ZMQ :5555---> [1 kHz joint impedance] --> Franka 172.16.0.2
        +-- gripper open/close -ZMQ :5559---> [RS-485 Modbus] ----------> Robotiq 2F-140
```

Execution is **open loop**: one plan per episode, streamed as cached waypoints.

## Step 0 — verify preconditions, report before installing

```bash
uname -a; uname -v                      # want PREEMPT_RT in the version string
ls /sys/kernel/realtime                 # exists on an RT kernel
grep -r rtprio /etc/security/limits.conf /etc/security/limits.d/ 2>/dev/null
groups                                  # 'realtime' group membership
ip -br addr                             # which NIC is on the 172.16.0.x subnet
ping -c3 172.16.0.2
ls -l /dev/ttyUSB*                      # Robotiq RS-485 adapter
locate libfranka.so 2>/dev/null || find / -name 'libfranka.so*' 2>/dev/null | head
lsb_release -a
```

**If there is no RT kernel, stop and report.** Installing Bamboo on a
non-RT kernel will build fine and then drop control-loop cycles, which the
Franka answers with `communication_constraints_violation` reflexes.

## Step 1 — libfranka version: 0.9.2. Do not go higher.

The installer will ask you for a libfranka version. The answer is **0.9.2**.

Two independent reasons, both checked against Franka's own docs:

1. **Compatibility matrix**: robot system version 4.2.2 falls in the
   `libfranka >= 0.9.1 <-> system >= 4.2.1` band. The next band up
   (`>= 0.10.0`) requires system `>= 5.2.0`, which this robot does not have.
2. **Panda support**: libfranka's CHANGELOG lists "Panda system version >= 5.2.0"
   for 0.10.0, and from **0.11.0 onwards the requirement is written as
   "Franka Research 3"** — Panda is no longer supported. A newer libfranka will
   simply fail to handshake with this arm.

**Pinocchio is NOT needed.** Bamboo's README requires it only for
libfranka >= 0.14.0. Skip that prerequisite entirely.

## Step 2 — install Bamboo

```bash
git clone https://github.com/chsahit/bamboo.git
cd bamboo
bash InstallBambooController      # answer 0.9.2 when prompted
```

Notes:
- It builds libfranka locally and does **not** overwrite system installs.
- It will ask for sudo to add user groups and install system packages
  (`libzmq3-dev`, `msgpack-dev`, `poco-dev`) — you must have the user approve
  these interactively; sudo needs a TTY.
- **If it adds you to any group, log out and back in before continuing.**

## Step 3 — Franka Desk

The user does this in a browser at `https://172.16.0.2/desk/`:

1. Unlock joints, release E-stop.
2. **Activate FCI** (Settings → ... → FCI). This only opens the control
   channel; it does not move anything. Bamboo is what drives it.
3. Set the robot to **Execution** mode (Programming mode is for hand-guiding).
4. Settings → **End-Effector**: enter the **Robotiq 2F-140** payload —
   mass, centre of mass, inertia tensor — and the TCP transform. Getting this
   wrong causes constant reflex stops under motion. The 2F-140 is heavier and
   longer than the 2F-85 in Robotiq's DROID docs, so use 2F-140 figures.
5. Restart the controller so the payload takes effect.

**Franka Desk will never show the Robotiq gripper.** Desk only knows the
original Franka Hand; third-party grippers are invisible to it and are driven
entirely over RS-485. That is expected, not a fault.

## Step 4 — the Robotiq 2F-140

The gripper is **not** powered or controlled through the Franka. It needs:

1. its own **24 V supply**,
2. an **RS-485 → USB adapter into this machine** (expect `/dev/ttyUSB0`),
3. an **activation command** over Modbus RTU, which Bamboo's gripper server
   sends on startup.

A **solid red LED means powered but not activated** (or a fault). It should go
solid blue once Bamboo activates it. If it stays red after Bamboo starts,
check 24 V, the tty enumeration, and the connector at the coupling.

Start Bamboo in **default** mode — that is the Robotiq path and it launches the
separate gripper server. Do **not** pass `--gripper_type franka`.

```bash
bash RunBambooController          # keep this running for the whole session
bash RunBambooController -h       # to set robot IP / gripper tty if not default
```

Success looks like: the gripper performs one open/close cycle and the LED
turns blue.

## Step 5 — verify

```bash
conda activate bamboo
python bamboo/examples/gripper.py          # NO --gripper-type flag (Robotiq)
```

Then, **only after clearing the space around the robot** — this script does no
collision checking and the arm must not be near a joint limit — and with the
user's hand on the E-stop:

```bash
python bamboo/examples/joint_trajectory.py
```

Stop with `bash RunBambooController stop`.

## Step 6 — report back to the user these facts

The GPU-side session needs all of them:

1. `PREEMPT_RT` present? exact `uname -v`.
2. libfranka version actually installed, and where.
3. This machine's **IP address on the network `zhiwei` can reach** — it goes
   into `robot.host` in `tiptop/config/tiptop.yml` on the GPU box.
4. Confirmation that Bamboo's robot port is **5555** and gripper port **5559**
   (or the ports you actually used).
5. The Robotiq tty path and whether the LED went blue.
6. Whether `joint_trajectory.py` and `gripper.py` both passed.
7. Anything the installer changed about groups, limits, or udev rules.

## Safety

- Hand on the E-stop for every first run of anything that moves the arm.
- Clear the workspace before `joint_trajectory.py`.
- The user is mid-way through a gripper swap (Franka Hand → Robotiq 2F-140).
  If the payload parameters in Desk still describe the old end-effector, the
  arm will behave badly. Confirm step 3.4 was done before any motion.
