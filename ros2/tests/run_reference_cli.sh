#!/usr/bin/env bash
# L3 - Reference smoke test. Runs ANSR directly (no ROS) on the tiny fixture and
# checks the output structure that the ROS retrieval service will later read.
#
# Runnable TODAY, before any ROS code exists. Run this first to confirm the venv
# and the fixture are sound - if this fails, nothing about the ROS layer is at fault.
#
# Usage:  ./run_reference_cli.sh [max_backprops]     (default 2000)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
SRC="$(cd "$ROOT/.." && pwd)/source_code"
MAX_BACKPROPS="${1:-2000}"

# --- Session workspace, laid out to satisfy the core's hard-coded path prefixes:
#     -t -> topologies/ , --train_data/--valid_data -> data/ , -c -> ./configs/
WORK="$(mktemp -d -t ansr-refrun-XXXXXX)"
mkdir -p "$WORK/data" "$WORK/topologies" "$WORK/results"
cp "$HERE/fixtures/data/smoke_train.csv"     "$WORK/data/"
cp "$HERE/fixtures/data/smoke_valid.csv"     "$WORK/data/"
cp "$HERE/fixtures/topologies/smoke_topology.txt" "$WORK/topologies/"

echo "== workspace: $WORK"
echo "== running ANSR (max_backprops=$MAX_BACKPROPS) ..."

# NOTE: bare filenames on purpose - the core prepends topologies/ and data/ itself.
# NOTE: cwd is the workspace, NOT source_code; Main.py is invoked by absolute path so
#       sys.path[0] still resolves the ANSR modules.
( cd "$WORK" && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python "$SRC/Main.py" \
    -t smoke_topology.txt \
    --train_data smoke_train.csv \
    --valid_data smoke_valid.csv \
    -o results \
    --maxTotalBackprops "$MAX_BACKPROPS" )

# --- Verify the output structure the ROS service depends on -------------------
echo
echo "== verifying output structure =="
fail() { echo "FAIL: $1"; echo "(workspace kept for inspection: $WORK)"; exit 1; }

SEED_DIR="$WORK/results/seed=1"
[ -d "$SEED_DIR" ] || fail "no seed dir at $SEED_DIR"

# The mirror is per-population and per-dataset-version: current_best_<version>
ARCHIVE_BEST="$(find "$SEED_DIR/archiveIndividuals" -maxdepth 1 -type d -name 'current_best_*' 2>/dev/null | sort -V | tail -1 || true)"
[ -n "$ARCHIVE_BEST" ] || fail "no current_best_* folder under $SEED_DIR/archiveIndividuals"

[ -f "$ARCHIVE_BEST/overview.txt" ] || fail "no overview.txt in $ARCHIVE_BEST"
[ "$(find "$ARCHIVE_BEST" -maxdepth 1 -name '*.txt' | wc -l)" -ge 2 ] || fail "expected >=1 individual .txt beside overview.txt"
[ "$(find "$ARCHIVE_BEST" -maxdepth 1 -name '*.m'   | wc -l)" -ge 1 ] || fail "expected >=1 individual .m"

echo "PASS - baseline output structure is as documented:"
echo "  $ARCHIVE_BEST"
echo "  files: $(find "$ARCHIVE_BEST" -maxdepth 1 -type f | wc -l)"
echo
echo "This is the exact tree the ROS get-model service must read."
echo "Workspace kept for inspection: $WORK"
