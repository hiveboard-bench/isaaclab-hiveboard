# Isaac Lab - HiveBoard Multi-Robot Manipulation Suite

[![Isaac Lab](https://img.shields.io/badge/IsaacLab-78b12aed-blue.svg)](https://github.com/isaac-sim/IsaacLab/commit/78b12aed1d2a56439493b8ea6d3538e6e379e2e5)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)](LICENSE)

The **[HiveBoard Benchmark](https://github.com/hiveboard-bench/HiveBoard)** for Isaac Lab,
simulated with Newton MJWarp (no Isaac Sim needed).

## Installation

```bash
git clone --recurse-submodules https://github.com/hiveboard-bench/isaaclab-hiveboard.git
cd isaaclab-hiveboard
just setup
```

Optional extras: `--extra curobo` (cuRobo-planned commands, needs CUDA),
`--extra cameras`, `--extra data`.

## Generate assets

The Newton USD assets are build outputs and must be generated once on a fresh clone.
URDF conversion uses [urdf-usd-converter](https://github.com/newton-physics/urdf-usd-converter);
`--uuc-python` points at a Python that can `import urdf_usd_converter`.

```bash
just verify-usd                                       # what is missing
just generate-usd --uuc-python /path/to/uuc/bin/python  # all HiveBoard assets and Spot
just generate-anymal-usd                              # ANYmal-D + DynaArm + Robotiq
just generate-honeycomb-collisions                    # panel colliders
```

## Layout

Robots, assets and single-goal tasks are data in `isaaclab_hiveboard/tasks_simple.py`, which also
builds the scene and the expert. The expert engine (sequential pose commands, skills, gripper
action, physics events) lives in `isaaclab_hiveboard/mdp/`.
