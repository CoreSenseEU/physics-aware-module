# physics_amm — ROS 2 wrapper for PhysicsAMM / ANSR

**PhysicsAMM / ANSR** is a symbolic-regression method that learns analytic formulas from data
(neuro-evolutionary search over network-shaped formula topologies with gradient-based
coefficient tuning). This repository wraps the unmodified ANSR core (`source_code/`)
as a ROS 2 module for the [CoreSense](https://coresense.eu) ecosystem:

- **`physics_amm`** (`ament_python`) — the wrapper node,
- **`physics_amm_msgs`** (`ament_cmake` + rosidl) — the interface definitions.

Learning runs are **decoupled from retrieval**: a learning run is a long-running,
cancellable **action**, while the current best-so-far models are always available through a
**service** that reads the core's on-disk `current_best_<version>/` mirror — including
*while* the run is still going, and *after* it has finished.

- **ROS 2 distro:** Jazzy Jalisco (Ubuntu 24.04). Default branch `rolling`, delivered as
  `jazzy-devel`.
- **License:** Apache-2.0. **Quality tier:** Tier 2.

## Video walkthrough

**[▶ Watch the 3-minute walkthrough](docs/walkthrough/walkthrough.mp4)** (GitHub opens it
in its built-in player).

The video is an unedited terminal recording of the *Build* and *Use* sections of this
README, executed in a clean environment (only idle waits are time-compressed). It shows:

1. **Build** — creating the venv, installing the ANSR stack, `colcon build` finishing
   with both packages, sourcing the workspace.
2. **Use** (three terminal panes: node / action / services) — launching the node with
   `python_executable` pointing at the venv; `start_session` returning a UUID; a
   `learn_model` goal on the bundled smoke fixtures with live feedback (validation loss
   converging within a minute); `get_model` returning an analytic model **while the run
   is still going**; `set_session_data` swapping the training data mid-run, with the
   dataset version advancing in the feedback; cancelling the goal (*cancel when
   satisfied*); `end_session` cleaning up.

The last `get_model` in the video returns `0.9997*x0**2 + 1.0004*x1` — the smoke
fixture's ground-truth formula (`y = x0² + x1`), learned live during the recording.

## Node

`physics_amm` (namespace `/physics_amm` by default; interface names below are relative to
the namespace).

### Services

| Name | Type | Purpose |
|---|---|---|
| `start_session` | `physics_amm_msgs/srv/StartSession` | Open a session; returns a UUID `session_id` |
| `end_session` | `physics_amm_msgs/srv/EndSession` | Close a session (`force:=true` cancels an active run first) |
| `get_model` | `physics_amm_msgs/srv/GetModel` | Current best-so-far model(s) from the on-disk mirror |
| `set_session_data` | `physics_amm_msgs/srv/SetSessionData` | Swap training (and optionally validation) data **mid-run** |

### Actions

| Name | Type | Purpose |
|---|---|---|
| `learn_model` | `physics_amm_msgs/action/LearnModel` | Run ANSR learning: long-running, cancellable, feedback every ~2 s |

The goal carries the session id and **exactly the five agreed ANSR parameters**; every
other ANSR setting takes its `SRConfig` default. Data crosses the interface **by file
path** (absolute paths on the node's host); the node stages copies in a private
per-session workspace.

| Goal field | ANSR switch | Required | Meaning |
|---|---|---|---|
| `topology_path` | `-t` | yes | master topology file |
| `train_data_path` | `--train_data` | yes | training dataset (CSV) |
| `valid_data_path` | `--valid_data` | yes | validation dataset (CSV) |
| `constraints_path` | `-c` | no (`""` = omitted) | prior-knowledge constraints file |
| `outfolder` | `-o` | yes | results directory name under the session workspace |

These five are the whole ANSR-facing surface — the subprocess command line contains
exactly these switches and nothing else (pinned by the L1 test in `tests/`). The full
field-level specification, including result codes and feedback fields, is in the comments
of `physics_amm_msgs/action/LearnModel.action` (`ros2 interface show
physics_amm_msgs/action/LearnModel`).

### Node parameters (defaults in `physics_amm/config/params.yaml`)

These configure the wrapper node itself — launch-time settings, not per-run inputs.
None of them is passed to ANSR; the five ANSR parameters above arrive with each
`learn_model` goal instead.

| Parameter | Default | Meaning |
|---|---|---|
| `workspace_root` | `~/.ros/physics_amm/sessions` | Per-session workspaces |
| `ansr_source_dir` | *(installed copy)* | Directory containing the ANSR `Main.py` |
| `python_executable` | *(node's interpreter)* | Interpreter for the ANSR subprocess — point this at the venv (see below) |
| `max_concurrent_runs` | `1` | Learning runs are serialized (each spawns a fixed 4-worker Dask tree) |
| `max_sessions` | `8` | Cap on open sessions |
| `sigterm_grace_s` | `10.0` | Grace before SIGTERM escalates to SIGKILL on cancel |
| `feedback_period_s` | `2.0` | Action feedback / cancellation poll rate |
| `max_run_seconds` | `36000.0` | Node-side wall-time watchdog (0 = off) |
| `default_population` | `archive` | Mirror `get_model` reads: `archive` (non-dominated) or `mainpop` |
| `mirror_read_retries` | `3` | Defensive re-reads of a mirror folder being rewritten |
| `cleanup_workspace_on_end` | `true` | Delete the workspace on `end_session` |

### Behaviour notes

- **Run termination.** The ANSR budget parameters are not exposed over ROS; a run ends when
  the core's own defaults are exhausted (50 000 backprops / 36 000 s), when the client
  **cancels** the action, or when the node's `max_run_seconds` watchdog fires. The natural
  client pattern is *cancel when satisfied* — `get_model` serves the best-so-far at any time.
- **Cancellation is process-group-wide.** ANSR spawns a Dask process tree and installs no
  signal handlers; the node launches it in its own session and cancels with
  `killpg` SIGTERM → SIGKILL, so no workers are orphaned.
- **Mirror staleness.** Mirror writes are throttled (5 s default), and each dump rewrites the
  folder; `get_model` may return state a few seconds old and retries partial reads. Right
  after a `set_session_data` swap the new dataset version's mirror is briefly sparse while
  individuals are re-measured.
- **Serialized runs.** A goal sent while another run is active is **rejected** (retryable),
  not queued.

## Build

Install the system prerequisites and create the virtual environment that will hold the
ANSR stack. The venv is created on the same interpreter ROS Jazzy uses (Python 3.12) with
system packages visible, and it is **never activated** in this workflow — it is used via
absolute paths (its pip below; its python at runtime, see *Use*). Run in: anywhere.

```bash
sudo apt install ros-jazzy-ros-base python3.12-venv python3-colcon-common-extensions
python3.12 -m venv --system-site-packages ~/venvs/ansr
```

Install the ANSR requirements and build. Run in: the **repository root** (the directory
containing `source_code/` and `ros2/`) — the first line captures its absolute path, after
which the remaining commands work from any directory in the same terminal. The second
line is a checkpoint: it must print the `Main.py` path; if it errors, you are not in the
repository root — `cd` there and rerun the block. The whole block is safe to re-run
(`ln -sfn` replaces the workspace symlink if it already exists):

```bash
REPO=$(pwd)
ls $REPO/source_code/Main.py
~/venvs/ansr/bin/pip install -r $REPO/source_code/requirements.txt
source /opt/ros/jazzy/setup.bash
mkdir -p ~/ros2_ws/src
ln -sfn $REPO/ros2 ~/ros2_ws/src/physics_amm_ros
cd ~/ros2_ws && colcon build --symlink-install
source install/setup.bash
```

The build should end with `Summary: 2 packages finished`. If it reports 0 packages, the
symlink target is wrong — inspect it with `readlink -f ~/ros2_ws/src/physics_amm_ros`.
The final `source` line is needed again in every new terminal — see *Use*.

`rclpy` comes from `/opt/ros/jazzy` (it cannot be pip-installed); the venv layers the ANSR
stack (PyTorch, Dask, SymPy, …) on top.

**One thing to configure at launch: the ANSR interpreter.** The node itself runs under the
system Python (that is where `rclpy` lives), but the ANSR subprocess it spawns needs the
venv with PyTorch/Dask/SymPy. Tell it which interpreter to use via the `python_executable`
launch argument (see *Use* below) — passing the venv's python by absolute path replaces
activating it. If you skip this, the first `learn_model` goal fails with a
`ModuleNotFoundError` for `torch` in its result message. The node logs the interpreter it
will use at startup (`ANSR interpreter: …`) — check that line to confirm.

## Use

Run in: anywhere — all commands in this section use absolute paths.

Every new terminal needs the workspace overlay first (it chains the ROS underlay, so this
one line is enough; a `Package 'physics_amm' not found` error means it is missing):

```bash
source ~/ros2_ws/install/setup.bash
```

Launch the node. The `params_file` argument is optional (installed defaults are used
without it):

```bash
ros2 launch physics_amm physics_amm.launch.py \
    python_executable:=$HOME/venvs/ansr/bin/python \
    params_file:=<your params.yaml>
```

Open a session; the response contains the `session_id` UUID used in all further calls:

```bash
ros2 service call /physics_amm/start_session physics_amm_msgs/srv/StartSession
```

Start learning. Paths must be absolute, on the node's host:

```bash
ros2 action send_goal /physics_amm/learn_model physics_amm_msgs/action/LearnModel \
  "{session_id: '<uuid>', topology_path: '<abs>/topology.txt',
    train_data_path: '<abs>/train.csv', valid_data_path: '<abs>/valid.csv',
    constraints_path: '', outfolder: 'results'}" --feedback
```

Query the best-so-far models — at any time, during or after the run:

```bash
ros2 service call /physics_amm/get_model physics_amm_msgs/srv/GetModel \
  "{session_id: '<uuid>', population: '', max_models: 3}"
```

Swap the training data mid-run; the search adapts and the dataset version advances:

```bash
ros2 service call /physics_amm/set_session_data physics_amm_msgs/srv/SetSessionData \
  "{session_id: '<uuid>', train_data_path: '<abs>/new_train.csv', valid_data_path: ''}"
```

Stop when satisfied: cancel the action (Ctrl-C on `send_goal`, or through the action
API), then close the session:

```bash
ros2 service call /physics_amm/end_session physics_amm_msgs/srv/EndSession \
  "{session_id: '<uuid>', force: true}"
```

Data format: headerless comma-separated CSV, target in the last column. The bundled smoke
fixtures (`tests/fixtures/data/`) follow this format: 60 training / 40 validation samples
with inputs drawn uniformly from `[-2, 2]` and **noise-free** targets `y = x0² + x1`
(`smoke_*.csv`) resp. `y = x0² + 2·x1` (`smoke_*_B.csv`) — exactly representable by the
bundled topology, so short runs converge to near-zero loss (details in
`tests/README.md`).

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Package 'physics_amm' not found` on `ros2 launch` | The workspace overlay is not sourced in this terminal. Run `source ~/ros2_ws/install/setup.bash` (needed in **every** new terminal). Check with `ros2 pkg prefix physics_amm`. |
| `The passed action type is invalid` (or an unknown service/message type) on `ros2 action send_goal` / `ros2 service call` | Same cause as above: without the overlay, the `physics_amm_msgs` interface types are unknown to the CLI. Source the overlay in this terminal. |
| `learn_model` result is `CODE_FAILED` with `ModuleNotFoundError: No module named 'torch'` | The ANSR subprocess ran under the system Python instead of the venv. Launch with `python_executable:=$HOME/venvs/ansr/bin/python` and confirm the node's startup log line `ANSR interpreter: …` points at the venv. |
| `colcon build` reports `Summary: 0 packages finished` (and the package is never found afterwards) | The workspace symlink is dangling — usually a relative `ln -s` target. Check `readlink -f ~/ros2_ws/src/physics_amm_ros`; recreate it with the `ln -sfn` line from *Build* (absolute `$REPO`). |
| `get_model` succeeds but returns no models early in a run | Not an error: the best-so-far export appears only after the initial population is evaluated, and refreshes are throttled (~5 s). Retry after a while; feedback's `num_models` shows when models exist. |

## Tests

See `tests/README.md` for the layer design (L1–L6) and the environment each layer needs.
Quick reference — the first three commands run in the repository root, the last one in
`~/ros2_ws`:

- L1/L2 contract tests, no ROS needed: `pytest ros2/tests/`
- L3 baseline, venv sourced, no ROS: `./ros2/tests/run_reference_cli.sh`
- L4–L6 integration, built+sourced workspace and venv: `pytest ros2/tests/ -m ros`
- Package unit + launch tests: `colcon test --packages-select physics_amm`

## Citation

If you use PhysicsAMM/ANSR in academic work, please cite the PhysicsAMM publications
(contact the authors for the current reference).

## Funding

This work is part of the [CoreSense](https://coresense.eu) project. CoreSense is funded by
the European Union's Horizon Europe research and innovation programme under grant agreement
No. 101070254. Views and opinions expressed are those of the authors only and do not
necessarily reflect those of the European Union. Neither the European Union nor the granting
authority can be held responsible for them.
