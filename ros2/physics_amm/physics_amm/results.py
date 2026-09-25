"""Reader for the ANSR on-disk current-best mirror.

ROS-agnostic. The core mirrors both populations to
<outfolder>/seed=1/<popdir>/current_best_<version>/, where <version> starts at
1 and increments on every training-data reload (older folders are kept as
snapshots). Each dump wipes the folder's files before rewriting them and
writes are throttled (saveCurrentBestMinIntervalS, default 5 s), so reads must
target the highest version present and tolerate briefly partial folders.
"""

import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

POPULATION_DIRS = {
    "archive": "archiveIndividuals",       # non-dominated best-so-far
    "mainpop": "mainPopulationIndividuals",  # live search population
}

_VERSION_RE = re.compile(r"current_best_(\d+)$")
_IND_ID_RE = re.compile(r"_(\d+)$")


@dataclass
class ModelRecord:
    """One individual from a current_best snapshot (mirrors AnalyticModel.msg)."""

    expression: str = ""
    simplified_expression: str = ""
    valid_loss: float = math.nan
    complexity: float = math.nan
    num_active_nodes: int = -1
    rmse_constr: float = math.nan
    dataset_version: int = 0
    population: str = "archive"
    txt_path: str = ""
    m_path: str = ""
    ind_id: int = -1
    age: str = ""
    backprop_iters: str = ""
    extra: dict = field(default_factory=dict)


class MirrorReader:
    """Defensive reader over one run's mirror tree (results_root = the -o dir)."""

    def __init__(self, results_root, seed: int = 1):
        self.results_root = Path(results_root)
        self.seed = seed

    def population_dir(self, population: str = "archive") -> Path:
        try:
            sub = POPULATION_DIRS[population]
        except KeyError:
            raise ValueError(
                f"unknown population {population!r}; expected one of "
                f"{sorted(POPULATION_DIRS)}"
            )
        return self.results_root / f"seed={self.seed}" / sub

    def latest_version(self, population: str = "archive"):
        """Highest current_best_<version> present, or None. Never hard-codes _1."""
        pop_dir = self.population_dir(population)
        if not pop_dir.is_dir():
            return None
        versions = []
        for entry in pop_dir.iterdir():
            m = _VERSION_RE.search(entry.name)
            if m and entry.is_dir():
                versions.append(int(m.group(1)))
        return max(versions) if versions else None

    def read_best(self, population: str = "archive", retries: int = 3,
                  retry_delay: float = 0.25):
        """Return (version, [ModelRecord...]) for the newest snapshot.

        Dumps wipe the folder before rewriting, so an empty or partially
        written folder is retried; after the retries are exhausted the reader
        returns what it has (possibly (version, [])) instead of raising —
        a best-so-far query must degrade, not fail.
        """
        for attempt in range(max(1, retries)):
            version = self.latest_version(population)
            if version is None:
                records = []
            else:
                records = self._read_snapshot(
                    self.population_dir(population) / f"current_best_{version}",
                    version, population,
                )
            if records:
                return version or 0, records
            if attempt < retries - 1:
                time.sleep(retry_delay)
        return version or 0, records

    def summary(self, population: str = "archive"):
        """(version, count, best_valid_loss, best_complexity) from overview only.

        Cheap enough for a feedback timer: reads a single small file, no
        per-individual joins. Values are NaN until a mirror exists.
        """
        version = self.latest_version(population)
        if version is None:
            return 0, 0, math.nan, math.nan
        folder = self.population_dir(population) / f"current_best_{version}"
        entries = self._parse_overview(folder / "overview.txt")
        if not entries:
            return version, 0, math.nan, math.nan
        best = min(
            entries.values(),
            key=lambda e: e.get("valid_loss", math.inf)
            if not math.isnan(e.get("valid_loss", math.nan)) else math.inf,
        )
        return (
            version,
            len(entries),
            best.get("valid_loss", math.nan),
            best.get("complexity", math.nan),
        )

    # -- internals ---------------------------------------------------------

    def _read_snapshot(self, folder: Path, version: int, population: str):
        overview = self._parse_overview(folder / "overview.txt")
        records = []
        try:
            txt_files = sorted(folder.glob("*.txt"))
        except OSError:
            return []
        for txt in txt_files:
            if txt.name == "overview.txt":
                continue
            m = _IND_ID_RE.search(txt.stem)
            ind_id = int(m.group(1)) if m else -1
            rec = ModelRecord(
                dataset_version=version, population=population,
                txt_path=str(txt), ind_id=ind_id,
            )
            m_file = txt.with_suffix(".m")
            if m_file.exists():
                rec.m_path = str(m_file)
            try:
                self._parse_individual(txt, rec)
            except OSError:
                continue  # wiped mid-read; the retry loop handles it
            meta = overview.get(ind_id, {})
            if math.isnan(rec.valid_loss):
                rec.valid_loss = meta.get("valid_loss", math.nan)
            if math.isnan(rec.complexity):
                rec.complexity = meta.get("complexity", math.nan)
            if rec.num_active_nodes < 0:
                nodes = meta.get("num_active_nodes", math.nan)
                rec.num_active_nodes = int(nodes) if not math.isnan(nodes) else -1
            if math.isnan(rec.rmse_constr):
                rec.rmse_constr = meta.get("rmse_constr", math.nan)
            records.append(rec)
        # Preserve the overview's ordering (complexity-sorted) when available.
        if overview:
            order = {ind_id: i for i, ind_id in enumerate(overview)}
            records.sort(key=lambda r: order.get(r.ind_id, len(order)))
        return records

    @staticmethod
    def _parse_overview(path: Path) -> dict:
        """overview.txt -> {ind_id: {metric: value}}, insertion-ordered."""
        try:
            text = path.read_text()
        except OSError:
            return {}
        entries = {}
        current = None
        for line in text.splitlines():
            header = re.fullmatch(r"ind_(\d+)", line.strip())
            if header and not line.startswith((" ", "\t")):
                current = {}
                entries[int(header.group(1))] = current
                continue
            if current is None or ":" not in line:
                continue
            key, _, value = line.strip().partition(":")
            key, value = key.strip(), value.strip()
            if key == "nb. of active nodes":
                current["num_active_nodes"] = _to_float(value)
            elif key in ("complexity", "valid_loss", "rmse_constr"):
                current[key] = _to_float(value)
        return entries

    @staticmethod
    def _parse_individual(path: Path, rec: ModelRecord):
        """Extract formulas and metrics from one individual's .txt dump."""
        text = path.read_text()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("Analytic formula in Python:"):
                rec.expression = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("Simplified analytic formula:"):
                rec.simplified_expression = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("nb. of active nodes:"):
                rec.num_active_nodes = int(_to_float(stripped.split(":", 1)[1]))
            elif stripped.startswith("complexity:"):
                rec.complexity = _to_float(stripped.split(":", 1)[1])
            elif stripped.startswith("valid_loss:"):
                rec.valid_loss = _to_float(stripped.split(":", 1)[1])
            elif stripped.startswith("rmse_constr:"):
                rec.rmse_constr = _to_float(stripped.split(":", 1)[1])
            elif stripped.startswith("Age:"):
                rec.age = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("Backprop iterations:"):
                rec.backprop_iters = stripped.split(":", 1)[1].strip()


def _to_float(value: str) -> float:
    try:
        return float(value.strip())
    except (ValueError, AttributeError):
        return math.nan
