"""Per-session workspace staging and ANSR argv construction.

ROS-agnostic (D5.1 guideline #1): importable and testable without rclpy.

The ANSR core silently prepends fixed directories to four of its five exposed
path arguments (topologies/ for -t, data/ for --train_data/--valid_data,
./configs/ for -c), so the wrapper passes bare filenames and runs the
subprocess with its working directory set to a workspace laid out to match.
"""

import os
import shutil
import tempfile
from pathlib import Path


class SessionWorkspace:
    """A per-session working directory laid out for the ANSR path prefixes.

    <session_dir>/
      data/         train + valid CSVs (copied, preserving basenames)
      topologies/   master topology file
      configs/      constraints JSON (only when constraints are used)
      results/      default target for -o
      logs/         subprocess stdout/stderr
    """

    def __init__(self, path: str):
        self.path = str(path)
        self.train_name = None
        self.valid_name = None
        self.topology_name = None
        self.constraints_name = None

    @classmethod
    def create(cls, root, train_data, valid_data, topology, constraints=None):
        root = Path(root).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        session_dir = Path(tempfile.mkdtemp(prefix="session_", dir=root))

        ws = cls(str(session_dir))
        (session_dir / "results").mkdir()
        (session_dir / "logs").mkdir()

        data_dir = session_dir / "data"
        data_dir.mkdir()
        # Copies, not symlinks: an in-place data swap must never mutate the
        # client's original file.
        ws.train_name = cls._stage(train_data, data_dir)
        ws.valid_name = cls._stage(valid_data, data_dir)

        topo_dir = session_dir / "topologies"
        topo_dir.mkdir()
        ws.topology_name = cls._stage(topology, topo_dir)

        if constraints:
            configs_dir = session_dir / "configs"
            configs_dir.mkdir()
            ws.constraints_name = cls._stage(constraints, configs_dir)

        return ws

    @staticmethod
    def _stage(src, dest_dir: Path) -> str:
        src = Path(src)
        if not src.is_file():
            raise FileNotFoundError(f"cannot stage {src}: not a file")
        dest = dest_dir / src.name
        shutil.copy2(src, dest)
        return src.name

    def swap_train_data(self, new_path):
        self._swap(new_path, self.train_name)

    def swap_valid_data(self, new_path):
        self._swap(new_path, self.valid_name)

    def _swap(self, new_path, staged_name: str):
        """Atomically replace a staged data file, preserving its name.

        The core's reload_data() watches the staged path's mtime; writing to a
        temp file in the same directory and os.replace()-ing keeps the watched
        path valid at every instant.
        """
        new_path = Path(new_path)
        if not new_path.is_file():
            raise FileNotFoundError(f"cannot swap in {new_path}: not a file")
        data_dir = Path(self.path) / "data"
        fd, tmp = tempfile.mkstemp(dir=data_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as out, open(new_path, "rb") as src:
                shutil.copyfileobj(src, out)
            os.replace(tmp, data_dir / staged_name)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def results_root(self, outfolder: str) -> str:
        """Absolute path of the -o directory for this workspace."""
        return str(Path(self.path) / outfolder)

    def remove(self):
        shutil.rmtree(self.path, ignore_errors=True)


def build_ansr_argv(*, topology, train_data, valid_data, outfolder,
                    constraints=None) -> list:
    """Build the ANSR CLI argument list — exactly the five agreed switches.

    Bare filenames only: the core prepends topologies/ and data/ (and
    ./configs/ for -c), so any directory component would break resolution.
    -c is omitted entirely when no constraints file is given (the parser
    defaults it to None).
    """
    argv = [
        "-t", os.path.basename(str(topology)),
        "--train_data", os.path.basename(str(train_data)),
        "--valid_data", os.path.basename(str(valid_data)),
        "-o", str(outfolder),
    ]
    if constraints:
        argv += ["-c", os.path.basename(str(constraints))]
    return argv
