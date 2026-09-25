"""Unit tests for MirrorReader against synthetic current_best trees.

The synthetic fixtures replicate the formats written by
PopulationManager._dump_current_best (overview.txt) and
Individual.saveIndividualPerformanceToFile (per-individual .txt/.m).
"""

import math
from pathlib import Path

from physics_amm.results import MirrorReader, ModelRecord  # noqa: F401

ARCHIVE = "archiveIndividuals"


def _write_individual(folder: Path, prefix: str, ind_id: int, *,
                      expr="x0**2 + x1", simple="x0**2 + x1",
                      valid_loss=0.001234, complexity=5.0, nodes=3):
    txt = folder / f"{prefix}_{ind_id}.txt"
    txt.write_text(
        "Performance metrics:\n"
        f"\tnb. of active nodes: {nodes}\n"
        f"\tcomplexity: {complexity}\n"
        f"\tvalid_loss: {valid_loss:2.15f}\n"
        f"\trmse_constr: {0.0:2.15f}\n"
        "\n\nAge:12\n"
        "\nBackprop iterations:340\n"
        "\nParents: 1, 2\n"
        "\nGenealogy: []\n"
        f"\n\nAnalytic formula in Python: {expr}\n"
        f"\n\nSimplified analytic formula: {simple}\n"
        f"\n\nAnalytic formula in MatLab: {expr.replace('**', '.^')}\n"
    )
    (folder / f"{prefix}_{ind_id}.m").write_text(f"y = {expr};\n")


def _write_overview(folder: Path, entries):
    lines = [f"# Snapshot of {len(entries)} individuals", "backprops: 1000", ""]
    for ind_id, metrics in entries:
        lines.append(f"ind_{ind_id}")
        lines.append(f"  nb. of active nodes: {metrics.get('nodes', 3)}")
        lines.append(f"  complexity: {metrics.get('complexity', 5.0)}")
        lines.append(f"  valid_loss: {metrics.get('valid_loss', 0.001):.15f}")
        lines.append(f"  rmse_constr: {metrics.get('rmse_constr', 0.0):.15f}")
        lines.append("  Age: 12")
        lines.append("  Backprop iterations: 340")
        lines.append("")
    (folder / "overview.txt").write_text("\n".join(lines))


def _make_snapshot(root: Path, version: int, ids=(7, 3)):
    folder = root / "seed=1" / ARCHIVE / f"current_best_{version}"
    folder.mkdir(parents=True)
    for ind_id in ids:
        _write_individual(folder, "archive_ind", ind_id,
                          valid_loss=0.001 * ind_id)
    _write_overview(folder, [(i, {"valid_loss": 0.001 * i}) for i in ids])
    return folder


def test_latest_version_picks_highest_not_first(tmp_path):
    _make_snapshot(tmp_path, 1)
    _make_snapshot(tmp_path, 3)
    _make_snapshot(tmp_path, 2)
    assert MirrorReader(tmp_path).latest_version("archive") == 3


def test_latest_version_none_before_first_dump(tmp_path):
    assert MirrorReader(tmp_path).latest_version("archive") is None
    version, records = MirrorReader(tmp_path).read_best(retries=1)
    assert version == 0 and records == []


def test_read_best_parses_expressions_and_metrics(tmp_path):
    _make_snapshot(tmp_path, 1, ids=(4,))
    version, records = MirrorReader(tmp_path).read_best("archive")
    assert version == 1
    assert len(records) == 1
    rec = records[0]
    assert rec.expression == "x0**2 + x1"
    assert rec.simplified_expression == "x0**2 + x1"
    assert abs(rec.valid_loss - 0.004) < 1e-12
    assert rec.num_active_nodes == 3
    assert rec.ind_id == 4
    assert rec.population == "archive"
    assert rec.txt_path.endswith("archive_ind_4.txt")
    assert rec.m_path.endswith("archive_ind_4.m")


def test_read_best_reads_newest_version_after_data_swap(tmp_path):
    _make_snapshot(tmp_path, 1)
    _make_snapshot(tmp_path, 2, ids=(9,))
    version, records = MirrorReader(tmp_path).read_best("archive")
    assert version == 2
    assert [r.ind_id for r in records] == [9]
    assert records[0].dataset_version == 2


def test_read_best_tolerates_wiped_folder(tmp_path):
    """A dump wipes files before rewriting; an empty folder must not raise."""
    folder = tmp_path / "seed=1" / ARCHIVE / "current_best_1"
    folder.mkdir(parents=True)
    version, records = MirrorReader(tmp_path).read_best(
        "archive", retries=2, retry_delay=0.01)
    assert version == 1
    assert records == []


def test_read_best_without_overview_falls_back_to_txt_files(tmp_path):
    folder = tmp_path / "seed=1" / ARCHIVE / "current_best_1"
    folder.mkdir(parents=True)
    _write_individual(folder, "archive_ind", 2)
    version, records = MirrorReader(tmp_path).read_best("archive")
    assert version == 1
    assert len(records) == 1
    assert records[0].expression == "x0**2 + x1"


def test_overview_nan_values_tolerated(tmp_path):
    folder = tmp_path / "seed=1" / ARCHIVE / "current_best_1"
    folder.mkdir(parents=True)
    (folder / "overview.txt").write_text(
        "# Snapshot of 1 individuals\nbackprops: 0\n\n"
        "ind_5\n  nb. of active nodes: nan\n  complexity: nan\n"
        "  valid_loss: nan\n  rmse_constr: nan\n"
    )
    txt = folder / "archive_ind_5.txt"
    txt.write_text("\n\nAnalytic formula in Python: x0\n")
    version, records = MirrorReader(tmp_path).read_best("archive")
    assert version == 1
    rec = records[0]
    assert rec.expression == "x0"
    assert math.isnan(rec.valid_loss)
    assert rec.num_active_nodes == -1


def test_summary_reports_best_loss_from_overview(tmp_path):
    _make_snapshot(tmp_path, 2, ids=(8, 2))  # losses 0.008, 0.002
    version, count, best_loss, best_complexity = MirrorReader(tmp_path).summary(
        "archive")
    assert (version, count) == (2, 2)
    assert abs(best_loss - 0.002) < 1e-12
    assert not math.isnan(best_complexity)


def test_summary_before_any_mirror(tmp_path):
    version, count, best_loss, _ = MirrorReader(tmp_path).summary("archive")
    assert (version, count) == (0, 0)
    assert math.isnan(best_loss)


def test_records_preserve_overview_order(tmp_path):
    """The overview is complexity-sorted by the core; GetModel keeps that order."""
    folder = tmp_path / "seed=1" / ARCHIVE / "current_best_1"
    folder.mkdir(parents=True)
    for ind_id in (10, 4, 7):
        _write_individual(folder, "archive_ind", ind_id)
    _write_overview(folder, [(4, {}), (10, {}), (7, {})])
    _, records = MirrorReader(tmp_path).read_best("archive")
    assert [r.ind_id for r in records] == [4, 10, 7]


def test_mainpop_population_dir(tmp_path):
    folder = tmp_path / "seed=1" / "mainPopulationIndividuals" / "current_best_1"
    folder.mkdir(parents=True)
    _write_individual(folder, "ind", 1)
    _write_overview(folder, [(1, {})])
    version, records = MirrorReader(tmp_path).read_best("mainpop")
    assert version == 1
    assert records[0].population == "mainpop"
