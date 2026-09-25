# Test plan — verifying the ROSified module against the bare Python package

## The key constraint: ANSR is not bit-reproducible

Do **not** build a golden-output test that compares expressions or loss values between a
reference run and a ROS run.

ANSR is an **asynchronous** evolutionary method driven by Dask `as_completed`, so the order in
which tuned individuals come back depends on worker scheduling. Two runs with the same seed and
the same data will not, in general, produce the same expression. A test asserting expression
equality — or even equal `valid_loss` — will be flaky and will be switched off within a week.

**Parity is therefore asserted on the contract and on invariants, not on exact outputs.**

That is not a weakness of the test suite. The thing that can actually break during ROSification
is the *plumbing* — wrong CLI arguments, wrong working directory, wrong output path, leaked
processes — and every one of those is deterministically testable.

## Test layers

| # | Layer | Needs ROS? | Speed | What it proves |
|---|---|---|---|---|
| L1 | Argv contract | no | ms | The node invokes ANSR exactly as a human would |
| L2 | Workspace staging | no | ms | Path prefixes are satisfied |
| L3 | Reference smoke | no | ~1 min | The fixture works; baseline output structure |
| L4 | ROS smoke | yes | ~1 min | The action produces the same structure as L3 |
| L5 | Extended mode | yes | ~2 min | Data swap + mid-run query work |
| L6 | Cancellation / no orphans | yes | ~30 s | **The Dask tree is fully reaped** |

**L1, L2 and L6 are the high-value tests.** L1/L2 catch the path-prefix class of bug that this
design is most exposed to, and they are fast and deterministic enough to run on every commit.
L6 covers the single most dangerous runtime failure (orphaned Dask workers eating a robot's CPU).

### L1 — Argv contract (unit)

Given a goal, assert the node builds exactly the expected invocation:

- exactly the five agreed switches, no more (`-t`, `-c`, `--train_data`, `--valid_data`, `-o`)
- `-c` omitted entirely when no constraints file is supplied
- **bare filenames** for `-t`, `--train_data`, `--valid_data` (never absolute paths — the core
  prepends `topologies/` and `data/`)
- `cwd` set to the session workspace
- `start_new_session=True`
- thread-limiting env vars present

No ANSR execution. This is the real "matches the original package" check.

### L2 — Workspace staging (unit)

Given a goal carrying arbitrary absolute paths, assert the staged session workspace is:

```text
<session_dir>/
  data/         smoke_train.csv, smoke_valid.csv
  topologies/   smoke_topology.txt
  configs/      <constraints>.json   (only when -c is used)
```

and that every staged file resolves (symlink or copy). Assert two sessions get disjoint
workspaces.

### L3 — Reference smoke (no ROS) — `run_reference_cli.sh`

Runs ANSR directly on the fixture with a **small `--maxTotalBackprops`** (possible here because
we are calling the CLI, not going through ROS) and asserts:

- exit code 0
- `<out>/seed=1/archiveIndividuals/current_best_1/` exists
- it contains `overview.txt` plus at least one `.txt` and one `.m`
- the reported `valid_loss` is finite

This establishes the baseline structure that L4 compares against, and is runnable **today**,
before any ROS code exists. Run it first to confirm the venv and fixture are sound.

### L4 — ROS smoke

Same fixture through the action. Assert: goal accepted → feedback received → the
`current_best_<version>/` tree appears at the documented path → the get-model service returns a
non-empty model.

**Terminate the run by cancelling it.** The budget parameters are not exposed, so a ROS-driven
run cannot be given a small backprop budget; it would otherwise run to 50 000 backprops.

Compare against L3 on **structure**: same directory layout, same file kinds, both produce a
model with finite loss. Not on values.

### L5 — Extended mode

1. Start a learning goal on `smoke_train.csv`.
2. Once `current_best_1/` appears, overwrite `<session>/data/smoke_train.csv` with
   `smoke_train_B.csv` (a different target function — `x0² + 2·x1` instead of `x0² + x1`).
3. Assert a `current_best_2/` folder appears → the dataset version advanced.
4. Call the get-model service *while the action is still running* → assert it returns a model
   (proves the multithreaded executor keeps services responsive).

Allow for the documented staleness: the mirror is throttled by `saveCurrentBestMinIntervalS`
(default 5 s), so poll with a timeout rather than asserting immediately.

### L6 — Cancellation / no orphans ⚠️

The most important integration test.

1. Start a goal; wait until Dask workers are up (poll for children of the subprocess).
2. Record the process-group id.
3. Cancel the action.
4. Assert that **within a few seconds no process remains in that group** —
   e.g. `os.killpg(pgid, 0)` raises `ProcessLookupError`, or `pgrep -g <pgid>` is empty.

Without process-group handling this test fails with ~8 orphaned processes still running. That
is exactly the failure mode this test exists to prevent.

## Running

```bash
# L1/L2 — default selection, no ROS or venv needed
pytest -v test_ros_parity.py

# L3 — venv only, no ROS (source the venv first)
./run_reference_cli.sh

# L4-L6 — need the venv AND a built+sourced workspace:
#   source .venv/bin/activate && source /opt/ros/jazzy/setup.bash && source install/setup.bash
pytest -v -m ros test_ros_parity.py
```

The `ros` and `slow` markers are deselected by default (see `pytest.ini`); the
integration tests spawn the node themselves via `ros_harness.py` — no separately
launched node is needed.

## Fixtures

Tiny by design so the suite is usable in CI:

| File | Target function | Rows |
|---|---|---|
| `data/smoke_train.csv` / `smoke_valid.csv` | `y = x0² + x1` | 60 / 40 |
| `data/smoke_train_B.csv` / `smoke_valid_B.csv` | `y = x0² + 2·x1` | 60 / 40 |
| `topologies/smoke_topology.txt` | 1 hidden layer: Multiply/Square/Ident ×2 | — |

Data generation: inputs `x0`, `x1` are sampled uniformly from `[-2, 2]`; the target is
computed **exactly** from the formula — the fixtures are deliberately **noise-free**
(verified: residuals against the target functions are identically zero). Noise-free
targets that are exactly representable by the fixture topology mean a short run converges
to a near-zero loss, which keeps the "did it learn anything at all?" assertion meaningful
without a long budget — and any non-trivial residual in a test points at the plumbing,
not at the data. CSVs are headerless, comma-separated, target in the last column,
matching `SRData`.
