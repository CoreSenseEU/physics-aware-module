"""Unit tests for the workspace staging details not covered by L1/L2:
the atomic in-place data swap and staging error paths."""

import pytest

from physics_amm.session import SessionWorkspace, build_ansr_argv


@pytest.fixture
def fixture_files(tmp_path):
    files = {}
    for name, content in (
        ("train.csv", "1,2,3\n"), ("valid.csv", "4,5,9\n"),
        ("topo.txt", "L1: Ident\n"), ("constr.json", "{}\n"),
        ("train_B.csv", "1,2,5\n"),
    ):
        p = tmp_path / name
        p.write_text(content)
        files[name] = p
    return files


def _create(tmp_path, files, constraints=None):
    return SessionWorkspace.create(
        root=tmp_path / "sessions",
        train_data=files["train.csv"], valid_data=files["valid.csv"],
        topology=files["topo.txt"], constraints=constraints,
    )


def test_swap_train_data_preserves_watched_path(tmp_path, fixture_files):
    ws = _create(tmp_path, fixture_files)
    staged = tmp_path / "sessions"
    staged_train = next(staged.glob("session_*")) / "data" / "train.csv"
    before = staged_train.read_text()

    ws.swap_train_data(fixture_files["train_B.csv"])

    assert staged_train.exists(), "watched path must survive the swap"
    assert staged_train.read_text() == "1,2,5\n"
    assert staged_train.read_text() != before
    # The client's original file is untouched (workspaces stage copies).
    assert fixture_files["train.csv"].read_text() == "1,2,3\n"
    # No temp files linger in data/.
    assert list(staged_train.parent.glob("*.tmp")) == []


def test_swap_missing_file_raises_and_keeps_staged_data(tmp_path, fixture_files):
    ws = _create(tmp_path, fixture_files)
    staged_train = next((tmp_path / "sessions").glob("session_*")) / "data" / "train.csv"
    with pytest.raises(FileNotFoundError):
        ws.swap_train_data(tmp_path / "does_not_exist.csv")
    assert staged_train.read_text() == "1,2,3\n"


def test_constraints_staged_into_configs_only_when_given(tmp_path, fixture_files):
    ws_without = _create(tmp_path, fixture_files)
    assert not (
        next((tmp_path / "sessions").glob(f"*{ws_without.path.split('_')[-1]}"))
        / "configs").exists()

    ws_with = _create(tmp_path, fixture_files,
                      constraints=fixture_files["constr.json"])
    from pathlib import Path
    assert (Path(ws_with.path) / "configs" / "constr.json").is_file()
    assert ws_with.constraints_name == "constr.json"


def test_create_rejects_missing_input(tmp_path, fixture_files):
    with pytest.raises(FileNotFoundError):
        SessionWorkspace.create(
            root=tmp_path / "sessions",
            train_data=tmp_path / "missing.csv",
            valid_data=fixture_files["valid.csv"],
            topology=fixture_files["topo.txt"],
        )


def test_workspace_remove(tmp_path, fixture_files):
    from pathlib import Path
    ws = _create(tmp_path, fixture_files)
    assert Path(ws.path).is_dir()
    ws.remove()
    assert not Path(ws.path).exists()


def test_argv_with_constraints_appends_c_flag():
    argv = build_ansr_argv(
        topology="/a/t.txt", train_data="/a/tr.csv", valid_data="/a/va.csv",
        outfolder="results", constraints="/a/cfg/constr.json")
    assert argv[argv.index("-c") + 1] == "constr.json"
