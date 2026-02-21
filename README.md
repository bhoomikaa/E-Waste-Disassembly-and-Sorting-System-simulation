# SafeDisassemble: E-Waste Disassembly & Sorting (Simulation + AI + Safety)

**One sentence:** SafeDisassemble is a virtual robot system that sees an electronic device (e.g., a laptop), identifies its components, plans a safe disassembly order, and removes each part in simulation—**especially avoiding dangerous battery damage**.

---

## Why this matters

Every year, ~**57 million tons** of electronics become e-waste. Much of it is not safely recycled, and manual disassembly can expose workers to toxic materials. Robots could help—but devices vary a lot (screws, clips, battery shape/placement, RAM/SSD layout), so we need systems that can **generalize** and **prioritize safety**.

---

## What this project does (high level)

SafeDisassemble is built like a three-level “organization”:

1. **Task Planner (CEO):** builds an ordered disassembly plan that respects dependencies and puts **battery disconnection first**.
2. **Skill Selector (Manager):** chooses *how* to execute each step (unscrew, pry, pull connector, lift component, etc.) and where to aim.
3. **Motor Policy (Worker):** generates smooth, frame-by-frame robot motion to execute the chosen skill.

A **Safety Module** validates the plan and continuously monitors forces near sensitive zones (battery and PCB) to pause/abort if risk rises.

---
## Repo Structure


ML Project/
├── safedisassemble/          ← The brain of the project
│   ├── models/               ← The AI that makes decisions
│   │   ├── task_planner/     ← "What should I do?" (Level 1 brain)
│   │   ├── skill_selector/   ← "How should I do it?" (Level 2 brain)
│   │   └── motor_policy/     ← "Move my hand exactly like this" (Level 3 brain)
│   ├── sim/                  ← The virtual world
│   │   ├── envs/             ← The rules of the world
│   │   ├── assets/xmls/      ← 3D models of the laptop, robot, table
│   │   └── device_registry.py ← Database of devices and their parts
│   ├── data/                 ← How to collect training examples
│   └── safety/               ← The safety guard
├── scripts/                  ← Tools to run demos and render videos
├── configs/                  ← Settings (like difficulty knobs)
└── tests/                    ← Checks that nothing is broken

---


## Part 1 — The Virtual World (Simulation)

### Physics Engine: MuJoCo
We use **MuJoCo** for realistic rigid-body dynamics, collisions, friction, and contacts—so actions like unscrewing, pulling a battery, or lifting a module behave physically.

### Devices (MJCF / XML models)
Devices are described in MJCF/XML “blueprints” with detailed geometry and materials.

- **Laptop**
  - chassis + back panel
  - **5 removable screws** (spin + lift joints)
  - **battery** (side-slide removal, ribbon connector)
  - **RAM module** (latch hinge + lift)
  - **SSD module** (side slide + mounting screw)
  - fan assembly (vertical lift)
  - motherboard / heatsink (present in the scene)
- **Router**
  - top shell with vents/logo
  - snap clips + hidden screws under feet
  - PCB, antenna connectors, CMOS coin cell

### Environment: `disassembly_env.py` (Gymnasium)
The environment is implemented using **Gymnasium**, so an agent can `reset()` and `step()` like standard RL tasks.

**Observations**
- 2 camera views (wrist + overhead), RGB **224×224**
- joint positions/velocities (7-DoF arm)
- gripper state
- end-effector pose + contact forces (estimated)

**Actions**
- 7 values: 6 for end-effector deltas (x/y/z + roll/pitch/yaw) + 1 for gripper open/close  
- normalized in **[-1, 1]**, scaled to safe step sizes

**Control**
- Jacobian-based end-effector control + PD stabilization.

### Visual “removal” animation
A joint-map (e.g., `_REMOVAL_JOINT_MAP`) defines how each component is removed:
- screws: rotate ~5 turns + lift out
- panel: slide upward
- battery: slide sideways
- RAM: open latch + lift

Removals animate smoothly (cubic easing) for realistic demos.

---

## Part 2 — The AI Brains (Models)

### Level 1: Task Planner (`task_planner/`)
- Takes device perception + instruction (e.g., “Disassemble this laptop”)
- Uses a vision-language approach (template retrieval + adaptation)
- Enforces the dependency graph via topological sorting
- **Hard rule:** battery handling/disconnection is prioritized before internal work

**Output:** ordered steps like:
- `unscrew -> screw_1`
- `pry_open -> back_panel`
- `pull_connector -> battery`
- `lift_component -> ssd_module`
- `lift_component -> ram_module`

### Level 2: Skill Selector (`skill_selector/`)
For each plan step, predicts:
- **Skill ID** (one of several manipulation primitives)
- Continuous parameters: target position, approach direction, max force, gripper width, rotation angle, confidence, etc.

Training uses:
- classification loss (skill)
- regression loss (parameters)

### Level 3: Motor Policy (`motor_policy/`)
We use a **Diffusion Policy** (conditional 1D U-Net) to generate motion trajectories.
Diffusion is useful because many manipulation tasks have **multiple valid trajectories** (multi-modal behavior), and diffusion avoids producing an “averaged” invalid motion.

---

## Part 3 — Safety System (`safety/`)

Safety is the core objective.

### Layer 1: Plan Validation
Before executing:
- verify dependency rules
- ensure battery is prioritized appropriately
- block/replan if a step is unsafe or inaccessible

### Layer 2: Real-Time Force Monitoring
At every control step:
- monitor proximity to safety zones (battery / PCB)
- smooth force readings
- respond with escalating actions:
  - **SAFE → CAUTION → WARNING → CRITICAL**
  - WARNING reduces speed
  - CRITICAL aborts and pauses

### Layer 3: Controller State Machine
System states:
`IDLE → PLANNING → EXECUTING → SAFETY_PAUSE/REPLANNING → COMPLETE/FAILED`

All safety events are logged.

---

## Part 4 — Data Pipeline (`data/`)

To generate large-scale training data, we use a scripted expert:
- builds valid plans from the dependency graph
- generates skill-specific waypoints (approach, engage, grip, unscrew, extract, transport, release)
- injects small noise for robustness
- stores episodes in **HDF5** with:
  - images, states, actions
  - instructions at multiple granularities
  - skill IDs
  - safety flags, success/failure, recovered components

---

## Part 5 — Training Configuration (`configs/default.yaml`)

Contains knobs for:
- visual augmentation (color jitter, noise, cutout)
- domain randomization (friction, mass, slight pose shifts)
- training schedules (epochs per module)
- evaluation settings (episodes, seen vs unseen device splits)

---

## Part 6 — Evaluation (`evaluation/`)

Metrics include:
- **Task completion rate**
- component recovery rate (and **value-weighted** recovery)
- safety violation rate
- battery puncture rate (target: **0%**)
- battery-first rate
- safe plan rate
- efficiency (steps per component)

---

## Part 7 — Cinematic Rendering (`scripts/cinematic_demo.py`)

Generates a documentary-style demo video:
- multiple cinematic camera presets + smooth interpolation
- 720p offscreen MuJoCo rendering
- automatic close-ups on active components
- overlays: titles, progress bar, lower-thirds
- H.264 encoding (ffmpeg), fallback to OpenCV encoder if needed

---
## Tech-Stack

| Area               | Technology                  | Why                                     |
| ------------------ | --------------------------- | --------------------------------------- |
| Physics simulation | MuJoCo                      | Accurate robotics physics + contacts    |
| Environment        | Gymnasium                   | Standard RL interface                   |
| Device modeling    | MJCF (XML)                  | Native MuJoCo format                    |
| Planning           | Vision-language + templates | Device understanding + structured plans |
| Skill selection    | CNN + Transformer + MLP     | Multi-modal decision making             |
| Motor control      | Diffusion Policy (1D U-Net) | Multi-modal trajectories                |
| Data storage       | HDF5                        | Efficient large datasets                |
| Video rendering    | MuJoCo renderer + ffmpeg    | High-quality demos                      |
| Language           | Python 3.11                 | ML/robotics ecosystem                   |


