import sys
import time
import copy
import math
import numpy as np
import os
import re
from collections import deque

import dask

from dask.distributed import Client, LocalCluster, as_completed, wait
from itertools import islice

import SRConfig as SRConfig
from Individual import (
    Individual,
    PerformanceHistory,
    splitPopulationByDomination,
    extractNondominatedSolutions,
    nonDominatedSorting,
    individualPerformanceSimilarity,
    latest_perf,
)
from MasterTopology import MasterTopology
from SRConstraints import SRConstraints
from SRData import SRData
from SRUnits import NNUnit
from SubTopology import SubTopology, getMutualDominance
from logger import log, log_population_state
from ParallelFinetuning import run_parallel_finetune
from TopologyInitializer import initialize_config_and_topologies
from final_model_selection import MCKPResult, select_mckp
from prefix import infix_to_prefix


def _patch_gc_diagnosis():
    """
    Call ONCE before the first LocalCluster() creation.
    Replaces enable/disable on the GCDiagnosis singleton with no-ops
    so dask's scheduler lifecycle works correctly across multiple clusters.
    """
    try:
        from distributed.gc import _gc_diagnosis

        _gc_diagnosis.enable = lambda: None
        _gc_diagnosis.disable = lambda: None
    except Exception:
        pass


_patch_gc_diagnosis()  # --- once here, before any LocalCluster is ever created


class _PopulationListView(list):
    """
    list subclass that invokes a callback after any operation that mutates the
    underlying sequence. Used by PopulationManager to keep `current_best/`
    on-disk snapshots of mainPopulation and archive in sync with their
    in-memory state at every change.

    Reads behave exactly like a plain list; only mutating operations differ
    (they fire `on_change` after the mutation has already been applied).
    Mutations covered: append, extend, insert, remove, pop, clear, sort,
    reverse, __setitem__, __delitem__, __iadd__. (__add__ returns a plain
    list, per Python convention - rebinding the attribute goes through
    PopulationManager's setter, which re-wraps.)
    """

    def __init__(self, iterable=(), on_change=None):
        super().__init__(iterable)
        self._on_change = on_change

    def _fire(self):
        if self._on_change is not None:
            try:
                self._on_change()
            except Exception as e:
                # Never let an on-disk mirror error break the EA loop.
                print(f"[_PopulationListView] on_change failed: {e!r}")

    def append(self, value):
        super().append(value); self._fire()

    def extend(self, iterable):
        super().extend(iterable); self._fire()

    def insert(self, index, value):
        super().insert(index, value); self._fire()

    def remove(self, value):
        super().remove(value); self._fire()

    def pop(self, index=-1):
        result = super().pop(index); self._fire(); return result

    def clear(self):
        super().clear(); self._fire()

    def sort(self, *args, **kwargs):
        super().sort(*args, **kwargs); self._fire()

    def reverse(self):
        super().reverse(); self._fire()

    def __setitem__(self, key, value):
        super().__setitem__(key, value); self._fire()

    def __delitem__(self, key):
        super().__delitem__(key); self._fire()

    def __iadd__(self, other):
        super().__iadd__(other); self._fire(); return self


class PopulationManager:
    # --- current_best/ mirror throttling. A burst of mutations (e.g. remove +
    #     append + sort in addOffspringToPopulation) within the configured
    #     minimum interval of the previous dump coalesces into a single
    #     deferred dump. Dirty populations are flushed by flush_pending_dumps(),
    #     which is called by the periodic save methods and at the end of
    #     runSymReg. The interval is read from SRConfig.saveCurrentBestMinIntervalS
    #     (default 5.0 s when the flag is absent).

    def __init__(self, masterTopology: MasterTopology, cli_args, timestamp):
        # --- Gate for the on-disk mirror of mainPopulation and archive.
        #     Kept False during __init__ so partially-constructed individuals
        #     (e.g. before age reset, before first tuning) are not written out.
        #     Flipped to True at the very end of __init__, where an initial
        #     dump is also triggered.
        self._dump_enabled = False
        # --- Throttling state for the current_best/ mirror.
        self._last_dump_main_time = 0.0
        self._last_dump_archive_time = 0.0
        self._dirty_main = False
        self._dirty_archive = False
        # --- Dataset-version counter. Starts at 1 for the dataset loaded at
        #     startup; bumped on every successful reload_data() return so each
        #     transition to a new training-data version gets its own
        #     current_best_<version>/ folder. The previous version's folder is
        #     left untouched, preserving the population state from when that
        #     dataset version was active.
        self._dataset_version = 1
        self.masterTopology = masterTopology
        self.cli_args = cli_args
        self.idCounter = 0
        self.archiveCounter = SRConfig.archiveIdMin
        self.lastPerturbationIdCounter = 0
        self.lastPerturbationIdBackprops = 0
        self.numOfBackpropTunings = 0
        self.start_time = SRConfig.start_time
        self.data_timestamp = timestamp
        self.queue = deque()
        self.totalBackpropsUsed = 0

        self.last_log_time = self.start_time
        self.last_log_backprops = 0
        self.last_log_ids = 0
        self.nb_tick_time = 0
        self.nb_tick_backprops = 0
        self.nb_tick_ids = 0

        # --- Process identification
        method = "ANSR" if not SRConfig.standard_EA else "seaSR"
        log(
            f"{method}, \nseed: {SRConfig.seed}\nTrain data: {SRConfig.train_data}\nValidation data: {SRConfig.valid_data}",
            start_time=self.start_time,
            file_path=SRConfig.log_file,
            level="INFO",
        )

        # --- Initialize max arities
        SRConfig.max_arities = {}
        self.last_arity_backprops = 0
        self.curr_arities_id: dict[str, int] = {}
        for unit_name, arity_schedule in SRConfig.arity_schedules.get(
            "values", {}
        ).items():
            SRConfig.max_arities[unit_name] = arity_schedule[0]
            self.curr_arities_id[unit_name] = 0
        log(
            "Initialized max arities: "
            + ", ".join(
                [
                    f"{unit_name}: {SRConfig.max_arities[unit_name]}"
                    for unit_name in SRConfig.max_arities
                ]
            ),
            start_time=self.start_time,
            file_path=SRConfig.log_file,
            level="HIGHLIGHT_RED",
        )

        # --- individuals from main population that are not being further tuned
        self.mainPopulationNotTuned: list[Individual] = []
        # --- individuals that are currently being tuned
        self.currentlyTuned: set[int] = set()
        # --- individuals that are not being tuned, but are waiting for others to be put back in the queue
        self.waitingNotTuned: dict[Individual, set[int]] = {}
        # --- dictionary of main population individual IDs and counter of times they were processed since the last merge of populations
        self.mainPopulationProcessed: dict[int, int] = {}
        # ---
        self.offspring: list[Individual] = []

        if SRConfig.initPopulationFromDirectory:
            self.mainPopulation: list[Individual] = self.initPopulationFromDirectory(
                SRConfig.initPopulationFromDirectory
            )
        elif SRConfig.initPopulationFromExpressions:
            self.mainPopulation: list[Individual] = self.initPopulationFromExpressions(
                SRConfig.initPopulationFromExpressions
            )
        else:
            self.mainPopulation: list[Individual] = self.initRandomPopulation(
                masterTopology, initPopulationSize=SRConfig.init_pop_size
            )

        # --- Initialize global mainPopulation worst performance values
        self.worst_valid: float = None
        self.worst_constr: float = None
        # _, _ = self.computeQuantilesFromMaximumPerformanceValues(population=self.mainPopulation)

        # --- Save individuals for geneaology analysis
        if SRConfig.genealogyPath.strip():
            for ind in self.mainPopulation:
                ind.saveIndividualToFile(
                    filePath=f"{SRConfig.genealogyPath}ind={ind.id}_backprops={ind.backprop_iters}.txt"
                )

        self.initialPopulationLearningPhaseI()
        # self.initialPopulationLearningPhaseI_combined()

        # --- Save individuals for geneaology analysis
        if SRConfig.genealogyPath.strip():
            for ind in self.mainPopulation:
                ind.saveIndividualToFile(
                    filePath=f"{SRConfig.genealogyPath}ind={ind.id}_backprops={ind.backprop_iters}.txt"
                )

        self.initialPopulationLearningPhaseII()
        self.mainPopulation = self._nds_dataset_aware(
            self.mainPopulation,
            finalSize=SRConfig.main_pop_size,
            criteria=SubTopology.keysPerformance,
        )
        _, _ = self.computeQuantilesFromMaximumPerformanceValues(
            population=self.mainPopulation
        )

        # --- Save individuals for geneaology analysis
        if SRConfig.genealogyPath.strip():
            for ind in self.mainPopulation:
                ind.saveIndividualToFile(
                    filePath=f"{SRConfig.genealogyPath}ind={ind.id}_backprops={ind.backprop_iters}.txt"
                )

        # --- Archive of the best-so-far solutions for each nbOfActiveNodes
        self.archive: list[Individual] = []
        # self.archive, _ = splitPopulationByDomination(self.mainPopulation, front=0, criteria=SubTopology.keysAll)
        self.updateArchive(newIndividuals=self.mainPopulation)
        log_population_state(
            population=self.archive,
            start_time=self.start_time,
            title="Archive",
            file_path=SRConfig.filePopulation,
        )

        # --- Reset age after initial learning phases
        for ind in self.mainPopulation:
            ind.age = SRConfig.totalItersInit
            ind.age_last_change_backprop = SRConfig.totalItersInit
            ind.backprop_iters = SRConfig.totalItersInit
            ind.adult = True if SRConfig.totalItersInit >= SRConfig.th_adult else False
            # log(f"ind[{ind.id}]backprop_iters: {ind.backprop_iters}", level="HIGHLIGHT_GREEN")

        # --- Install the global Individual.last_perf hook so the mirror also
        #     refreshes whenever a tracked individual's measured performance
        #     changes (e.g. after process_individual merges a worker result).
        Individual._on_perf_change_hook = self._on_individual_perf_change

        # --- Activate the on-disk mirror of mainPopulation and archive.
        #     From now on, any mutation of either list, any rebinding of the
        #     attribute, or any last_perf reassignment on a tracked individual
        #     triggers a write of <outputNamePrefix><folder>/current_best/
        #     (subject to the throttle and the saveCurrentBest SRConfig flag).
        self._dump_enabled = True
        # Force an unthrottled initial dump so the folders exist immediately.
        self._do_dump_current_best_main_population()
        self._do_dump_current_best_archive()

    # ------------------------------------------------------------------ #
    # mainPopulation and archive are exposed as properties so that BOTH  #
    # in-place mutations (.append/.remove/.sort/...) and rebindings      #
    # (self.mainPopulation = nonDominatedSorting(...)) keep the on-disk  #
    # mirror in sync. Reassignments re-wrap the new list in              #
    # _PopulationListView; in-place mutations fire the callback directly.#
    # ------------------------------------------------------------------ #

    @property
    def mainPopulation(self):
        return self._mainPopulation

    @mainPopulation.setter
    def mainPopulation(self, value):
        self._mainPopulation = _PopulationListView(
            value, on_change=self._dump_current_best_main_population
        )
        if getattr(self, "_dump_enabled", False):
            self._dump_current_best_main_population()

    @property
    def archive(self):
        return self._archive

    @archive.setter
    def archive(self, value):
        self._archive = _PopulationListView(
            value, on_change=self._dump_current_best_archive
        )
        if getattr(self, "_dump_enabled", False):
            self._dump_current_best_archive()

    # ------------------------------------------------------------------ #
    # current_best dumpers                                                #
    #                                                                     #
    # Public/observer-facing entry points are the "_dump_current_best_*"  #
    # methods - they honor SRConfig.saveCurrentBest, the _dump_enabled    #
    # gate, and the MIN_DUMP_INTERVAL_S throttle. The "_do_dump_..."      #
    # forms perform the actual disk writes unconditionally; they are      #
    # called by the throttled entries when the interval has elapsed and  #
    # by flush_pending_dumps() to discharge deferred dumps.               #
    # ------------------------------------------------------------------ #

    def _dump_current_best_main_population(self):
        """Throttled request to mirror mainPopulation. Sets _dirty_main if
        the dump is deferred because the previous one was too recent."""
        if not getattr(self, "_dump_enabled", False):
            return
        if not getattr(SRConfig, "saveCurrentBest", True):
            return
        now = time.time()
        if (
            now - self._last_dump_main_time
            >= getattr(SRConfig, "saveCurrentBestMinIntervalS", 5.0)
        ):
            self._do_dump_current_best_main_population(now)
        else:
            self._dirty_main = True

    def _dump_current_best_archive(self):
        """Throttled request to mirror archive. See _dump_current_best_main_population."""
        if not getattr(self, "_dump_enabled", False):
            return
        if not getattr(SRConfig, "saveCurrentBest", True):
            return
        now = time.time()
        if (
            now - self._last_dump_archive_time
            >= getattr(SRConfig, "saveCurrentBestMinIntervalS", 5.0)
        ):
            self._do_dump_current_best_archive(now)
        else:
            self._dirty_archive = True

    def _do_dump_current_best_main_population(self, now=None):
        """Unconditional disk write of mainPopulation. Updates throttling
        state. Use _dump_current_best_main_population unless you specifically
        need to bypass the throttle (initial dump, deferred flush)."""
        self._dump_current_best(
            population=self._mainPopulation,
            folder_name="mainPopulationIndividuals",
            file_prefix="ind",
        )
        self._last_dump_main_time = now if now is not None else time.time()
        self._dirty_main = False

    def _do_dump_current_best_archive(self, now=None):
        """Unconditional disk write of archive. See _do_dump_current_best_main_population."""
        self._dump_current_best(
            population=self._archive,
            folder_name="archiveIndividuals",
            file_prefix="archive_ind",
        )
        self._last_dump_archive_time = now if now is not None else time.time()
        self._dirty_archive = False

    def flush_pending_dumps(self):
        """Flush any deferred current_best/ dumps.

        Mutations or perf changes that hit the throttle window only mark the
        affected population as dirty. Without an explicit flush they would
        remain stale on disk until the next non-throttled mutation. This is
        called automatically by the periodic save methods
        (updateLogFileAndSaveModelsByTime/Backprops/Ids) and once at the end
        of runSymReg, so the on-disk mirror never lags behind by more than
        one save interval (or one final write).
        """
        if not getattr(self, "_dump_enabled", False):
            return
        if not getattr(SRConfig, "saveCurrentBest", True):
            # Feature disabled: clear dirty bits so we don't keep flagging.
            self._dirty_main = False
            self._dirty_archive = False
            return
        if self._dirty_main:
            self._do_dump_current_best_main_population()
        if self._dirty_archive:
            self._do_dump_current_best_archive()

    def _on_individual_perf_change(self, ind):
        """Hook installed on Individual._on_perf_change_hook. Fires whenever
        any Individual's last_perf is reassigned. Refreshes the on-disk
        mirror of whichever tracked population the individual currently
        belongs to. Membership is checked against the live wrapped lists so
        an individual that has been removed since does NOT cause a write.

        Cost: O(|mainPopulation| + |archive|) per perf change (two list
        membership tests). With default pop sizes (~20) and small archives
        this is microseconds. Throttling further coalesces bursts of perf
        changes (e.g. a batch of Dask results returning together).
        """
        if not getattr(self, "_dump_enabled", False):
            return
        if not getattr(SRConfig, "saveCurrentBest", True):
            return
        # Individual.__eq__ compares by id so 'in' is an identity-by-id check.
        if ind in self._mainPopulation:
            self._dump_current_best_main_population()
        if ind in self._archive:
            self._dump_current_best_archive()

    def _dump_current_best(self, population, folder_name, file_prefix):
        """
        Mirror `population` into <outputNamePrefix><folder_name>/current_best/.
        Wipes the folder first (so individuals removed since the last dump do
        not linger as stale files), then writes one .txt + one .m per
        individual via Individual.saveIndividualPerformanceToFile, and finally
        one overview.txt summarizing every individual in the folder.

        Gated by:
          * self._dump_enabled - False during __init__ so a partially
            constructed PopulationManager does not try to dump.
          * SRConfig.saveCurrentBest - master enable/disable flag for the
            feature. Defaults to True if the flag is absent from SRConfig.
        """
        if not getattr(self, "_dump_enabled", False):
            return
        if not getattr(SRConfig, "saveCurrentBest", True):
            return

        # --- Per-dataset-version folder. Each call to reload_data() bumps
        #     self._dataset_version; dumps that follow land in a fresh folder,
        #     while the previous folder is preserved as a snapshot of the
        #     population state under the previous dataset version.
        version = getattr(self, "_dataset_version", 1)
        folder_path = os.path.join(
            SRConfig.outputNamePrefix, folder_name, f"current_best_{version}"
        )
        try:
            os.makedirs(folder_path, exist_ok=True)
        except OSError as e:
            print(f"[_dump_current_best] cannot create {folder_path}: {e!r}")
            return

        # Wipe stale files from the previous snapshot (top-level files only;
        # we never put subfolders here).
        try:
            for entry in os.listdir(folder_path):
                entry_path = os.path.join(folder_path, entry)
                if os.path.isfile(entry_path):
                    try:
                        os.remove(entry_path)
                    except OSError:
                        pass
        except OSError:
            pass

        # Snapshot the population so concurrent mutation (shouldn't happen,
        # we're single-threaded, but cheap insurance) doesn't trip us mid-iter.
        snapshot = list(population)

        # Write each individual via the existing helper. With
        # tickValue=f"current_best_{version}" and prefix="", the resolved path
        # is exactly the folder we just cleaned:
        # <outputNamePrefix><folder_name>/current_best_<version>/.
        tick_value = f"current_best_{version}"
        for ind in snapshot:
            try:
                ind.saveIndividualPerformanceToFile(
                    folderName=folder_name,
                    fileName=f"{file_prefix}_{ind.id}",
                    tickValue=tick_value,
                    prefix="",
                )
            except Exception as e:
                print(
                    f"[_dump_current_best] failed to write ind "
                    f"{getattr(ind, 'id', '?')} into {folder_path}: {e!r}"
                )

        self._write_overview_for_population(
            folder_path,
            snapshot,
            sort_by_complexity=(folder_name == "archiveIndividuals"),
        )

    def _write_overview_for_population(self, folder_path, population, sort_by_complexity=False):
        """
        Write overview.txt summarizing every individual in `population`.
        Header lines record the snapshot size and the manager's
        totalBackpropsUsed at the moment of the dump (so a directory listing
        alone tells you which point in the run the snapshot belongs to).
        Per-individual fields: nb. of active nodes, complexity, valid_loss,
        rmse_constr, Age, Backprop iterations. Missing fields are written as
        'nan' rather than raising, so a partly-initialised individual cannot
        kill the dump.

        When sort_by_complexity is True (used for archive overviews), the
        listing is ordered by complexity ascending, ties broken by valid_loss
        ascending, with missing/NaN/non-numeric values sorted last on each key.
        Sorting is done on a local copy so the caller's list (e.g. self.archive,
        whose order matters to the EA) is never mutated.
        """
        # Always work on a copy: never reorder the caller's list in place.
        population = list(population)

        if sort_by_complexity:
            def _metric_key(value):
                # (is_missing, value): missing/NaN/non-numeric sort last.
                try:
                    f = float(value)
                except (TypeError, ValueError):
                    return (1, 0.0)
                if math.isnan(f):
                    return (1, 0.0)
                return (0, f)

            def _sort_key(ind):
                perf = (
                    getattr(getattr(ind, "last_perf", None), "performance", None)
                    or {}
                )
                return (
                    _metric_key(perf.get("complexity")),
                    _metric_key(perf.get("valid_loss")),
                )

            population.sort(key=_sort_key)

        lines = [
            f"# Snapshot of {len(population)} individuals",
            f"backprops: {int(getattr(self, 'totalBackpropsUsed', 0))}",
            "",
        ]
        for ind in population:
            perf = (
                getattr(getattr(ind, "last_perf", None), "performance", None) or {}
            )
            nb_active = perf.get("nbOfActiveNodes", float("nan"))
            try:
                nb_active_str = str(int(nb_active))
            except (TypeError, ValueError):
                nb_active_str = str(nb_active)
            complexity = perf.get("complexity", float("nan"))
            valid_loss = perf.get("valid_loss", float("nan"))
            rmse_constr = perf.get("rmse_constr", float("nan"))

            lines.append(f"ind_{ind.id}")
            lines.append(f"  nb. of active nodes: {nb_active_str}")
            lines.append(f"  complexity: {complexity}")
            lines.append(f"  valid_loss: {valid_loss:.15f}" if isinstance(valid_loss, float) else f"  valid_loss: {valid_loss}")
            lines.append(f"  rmse_constr: {rmse_constr:.15f}" if isinstance(rmse_constr, float) else f"  rmse_constr: {rmse_constr}")
            lines.append(f"  Age: {getattr(ind, 'age', 'n/a')}")
            lines.append(f"  Backprop iterations: {getattr(ind, 'backprop_iters', 'n/a')}")
            lines.append("")

        try:
            with open(os.path.join(folder_path, "overview.txt"), "w") as f:
                f.write("\n".join(lines))
        except OSError as e:
            print(f"[_write_overview_for_population] failed: {e!r}")

    def _write_population_overview(self, population, folderName, tickValue, prefix="time_"):
        """Write overview.txt into the same folder that
        Individual.saveIndividualPerformanceToFile uses for the given
        (folderName, tickValue, prefix):
            <outputNamePrefix><folderName>/<prefix><tickValue>/overview.txt

        Reuses the exact path formula from saveIndividualPerformanceToFile so
        the overview lands beside the per-individual .txt/.m files. Skips empty
        populations: the per-individual save loops only create the
        <prefix><tickValue>/ folder when the population is non-empty, so calling
        this right after such a loop guarantees the folder already exists (no
        makedirs needed) and we never create a folder that holds only an
        overview.txt (e.g. for an archive that is still empty early in a run).
        """
        population = list(population)
        if not population:
            return
        folder_path = SRConfig.outputNamePrefix + folderName + f"/{prefix}{tickValue}/"
        self._write_overview_for_population(
            folder_path,
            population,
            sort_by_complexity=(folderName == "archiveIndividuals"),
        )

    # ------------------------------------------------------------------ #
    # Dataset-aware comparison helpers                                    #
    #                                                                     #
    # Centralised enforcement of the rule "an individual whose last_perf  #
    # was measured against the manager's current data always strictly     #
    # dominates one whose last_perf was measured against an older data    #
    # version". Every dominance, NDS, and tournament selection site that  #
    # needs to respect dataset boundaries routes through these helpers    #
    # so the rule is structural, not advisory.                            #
    # ------------------------------------------------------------------ #

    def _is_perf_current(self, perf) -> bool:
        """True iff the given PerformanceHistory was measured against the
        manager's current data. Exact == on dataTimestamp: both sides come
        from os.path.getmtime of the same file, so identical values are
        bit-identical floats (no tolerance needed)."""
        return perf is not None and perf.dataTimestamp == self.data_timestamp

    def _has_current_perf(self, ind) -> bool:
        """True iff ind.last_perf was measured against the manager's current
        data. Convenience wrapper for callers that work with Individuals."""
        return self._is_perf_current(getattr(ind, "last_perf", None))

    def _dataset_aware_dominance(self, perf1, perf2, criteria):
        """Pairwise dominance with the 'current dataset always wins' rule.

        Drop-in replacement for getMutualDominance at every call site that
        needs to respect dataset boundaries. Takes PerformanceHistory
        objects (not Individuals) so callers comparing last_perf against a
        perf_history[age] entry - each side carrying its own dataTimestamp -
        can use the helper directly.

        Returns the same (firstDominates, secondDominates, firstEqualsSecond)
        tuple as getMutualDominance.

        Decision table:
            perf1 current, perf2 stale  -> (True,  False, False)  perf1 wins
            perf1 stale,   perf2 current-> (False, True,  False)  perf2 wins
            either is None              -> (False, False, False)  incomparable
            both stale                  -> (False, False, False)  incomparable
            both current                -> delegate to getMutualDominance
        """
        c1 = self._is_perf_current(perf1)
        c2 = self._is_perf_current(perf2)
        if c1 and not c2:
            return (True, False, False)
        if c2 and not c1:
            return (False, True, False)
        if perf1 is None or perf2 is None or (not c1 and not c2):
            # Both stale (or either missing): cross-dataset measurements are
            # not comparable on the underlying metrics, so neither dominates.
            return (False, False, False)
        return getMutualDominance(
            first=perf1.performance,
            second=perf2.performance,
            criteria=criteria,
        )

    def _nds_dataset_aware(self, population, finalSize, criteria):
        """nonDominatedSorting wrapper that respects 'current always wins'.

        Because each stale individual is dominated by every current one under
        the dataset rule, the resulting front order is:
            front 0..k from NDS(currents),
            front k+1..n from NDS(stales).
        Pre-partitioning gives the same result without editing
        nonDominatedSorting itself, and trivially honors finalSize - currents
        fill the first slots, stales fill any remainder.
        """
        currents = [ind for ind in population if self._has_current_perf(ind)]
        stales = [ind for ind in population if not self._has_current_perf(ind)]

        if len(currents) >= finalSize:
            return nonDominatedSorting(
                currents, finalSize=finalSize, criteria=criteria
            )

        currents_sorted = (
            nonDominatedSorting(
                currents, finalSize=len(currents), criteria=criteria
            )
            if currents
            else []
        )
        remaining = finalSize - len(currents_sorted)
        stales_sorted = (
            nonDominatedSorting(
                stales, finalSize=remaining, criteria=criteria
            )
            if stales and remaining > 0
            else []
        )
        return currents_sorted + stales_sorted

    def _tournament_pool(self, pop):
        """Restrict a tournament's sampling pool to currents when any exist.

        Equivalent to 'current always beats stale' inside the tournament: if
        even one current individual is in the original pool, the winner is
        sampled from currents only. If every member is stale (the brief
        window right after a reload, before any recompute has finished), the
        full pool is returned - the tournament still has to pick someone,
        and the best-of-stales is the best we can do until a recompute
        produces a current individual."""
        currents = [ind for ind in pop if self._has_current_perf(ind)]
        return currents if currents else list(pop)

    def update_max_arities(self):
        """
        Updates the maximum arities of units according to the user-defined schedule.
        """
        if (
            self.totalBackpropsUsed - self.last_arity_backprops
            >= SRConfig.arity_schedules.get("backprop_period", 1e9)
        ):
            self.last_arity_backprops = self.totalBackpropsUsed
            for unit_name, arity_schedule in SRConfig.arity_schedules.get(
                "values", {}
            ).items():
                self.curr_arities_id[unit_name] = min(
                    self.curr_arities_id[unit_name] + 1, len(arity_schedule) - 1
                )
                SRConfig.max_arities[unit_name] = arity_schedule[
                    self.curr_arities_id[unit_name]
                ]
            log(
                "Updated max arities: "
                + ", ".join(
                    [
                        f"{unit_name}: {SRConfig.max_arities[unit_name]}"
                        for unit_name in SRConfig.max_arities
                    ]
                ),
                start_time=self.start_time,
                file_path=SRConfig.log_file,
                level="HIGHLIGHT_CYAN",
            )

    def updateArchive(self, newIndividuals: list[Individual]):
        """
        Updates 'archive' with non-dominated solutions of 'newIndividuals'.

        Filters incoming individuals so only those whose last_perf was measured
        against the dataset currently held by the manager are considered. This
        is what enforces "only individuals tuned/evaluated on the current data
        can enter the archive": in steady state every individual already meets
        the condition (so behaviour is unchanged), and right after a data
        reload it drops any in-flight worker results that finished against the
        OLD data version before the reload was observed by the main process.
        Such individuals will pass the filter on a future call once their
        performance has been recomputed on the new data (in
        _handle_not_subject_to_tune or via the next worker round).
        """
        current_ts = self.data_timestamp
        admitted = []
        dropped_stale = 0
        for ind in newIndividuals:
            perf = ind.last_perf
            if perf is None:
                continue
            if perf.performance["nbOfActiveNodes"] < SRConfig.minNbOfActiveNodes:
                continue
            # Exact == is safe: both sides come from os.path.getmtime of the
            # same file, so identical values are bit-identical floats. See the
            # earlier comments in _handle_not_subject_to_tune / reload_data.
            if perf.dataTimestamp != current_ts:
                dropped_stale += 1
                continue
            admitted.append(ind)
        self.archive.extend(admitted)
        if dropped_stale:
            log(
                f"updateArchive: dropped {dropped_stale} individual(s) "
                f"whose last_perf predates the current data "
                f"(current data_timestamp={current_ts}, "
                f"dataset version={getattr(self, '_dataset_version', 1)})",
                start_time=self.start_time,
                file_path=SRConfig.log_file,
                level="DEBUG",
            )
        self.archive, self.archiveCounter = extractNondominatedSolutions(
            self.archive, self.archiveCounter
        )

        # --- Keep only the extreme solutions per number of active nodes
        # self.archive.extend([ind for ind in newIndividuals if ind.last_perf.performance["nbOfActiveNodes"] >= SRConfig.minNbOfActiveNodes])
        # self.archive, self.archiveCounter = extractNondominatedSolutionsPerActiveNodes(self.archive, self.archiveCounter)

    def tournamentSelectionRMSE(
        self,
        input_population: list[Individual],
        other: Individual = None,
        size: int = 1,
    ) -> Individual:
        # Dataset-aware: sample from currents only when any exist. Equivalent
        # to "current always wins" inside the tournament; falls back to the
        # full pool when every member is stale (brief post-reload window).
        input_population = self._tournament_pool(input_population)
        i = 1
        winner = input_population[SRConfig.r.integers(0, len(input_population))]
        while i < size:
            cand = input_population[SRConfig.r.integers(0, len(input_population))]
            # latest_perf falls back to last_perf when perf_history is empty
            # (the state every individual is in immediately after a reload,
            # before any recompute lands). float("inf") on a missing perf
            # makes the candidate the worst possible loss, so the existing
            # winner is preferred - degrades gracefully.
            cand_perf = latest_perf(cand)
            winner_perf = latest_perf(winner)
            last_cand_loss = (
                cand_perf.performance["valid_loss"]
                if cand_perf is not None
                else float("inf")
            )
            last_winner_loss = (
                winner_perf.performance["valid_loss"]
                if winner_perf is not None
                else float("inf")
            )

            if last_cand_loss < last_winner_loss:
                winner = cand
            i += 1
        return winner

    def tournamentSelectionFront(
        self,
        input_population: list[Individual],
        other: Individual = None,
        size: int = 1,
    ) -> Individual:
        # Dataset-aware: see tournamentSelectionRMSE.
        input_population = self._tournament_pool(input_population)
        i = 1
        winner = input_population[SRConfig.r.integers(0, len(input_population))]
        while i < size:
            cand = input_population[SRConfig.r.integers(0, len(input_population))]

            if cand.front < winner.front:
                winner = cand
            i += 1
        return winner

    def tournamentSelectionUniqueGenealogy(
        self,
        input_population: list[Individual],
        other: Individual = None,
        size: int = 1,
    ) -> Individual:
        """
        Implements a tournament selection, where it selects the best candidate among 'size' randomly selected ones from the 'input_population'.
        The winner is the one that has the least overlap of its 'linearized_genealogy' with 'linearized_genealogy' of the 'other' Individual.
        The 'linearized_genealogy' is a set of '<int1>/<int2>' without replacement extracted from the individual's 'genealogy' dictionary.
        """

        def linearized_genealogy(ind: Individual) -> set:
            # --- Flatten all tuples in genealogy.values() into one set of id/iters strings
            return {entry for entries in ind.genealogy.values() for entry in entries}

        other_lin = linearized_genealogy(other) if other is not None else set()

        # Dataset-aware: sample from currents only when any exist.
        input_population = self._tournament_pool(input_population)
        i = 1
        # print(f"\ntournamentSelectionUniqueGenealogy")
        winner = input_population[SRConfig.r.integers(0, len(input_population))]
        winner_overlap = len(linearized_genealogy(winner) & other_lin)
        # print(f"\tcand_overlap: {winner_overlap}")
        while i < size:
            cand = input_population[SRConfig.r.integers(0, len(input_population))]
            cand_overlap = len(linearized_genealogy(cand) & other_lin)
            # print(f"\tcand_overlap: {cand_overlap}")
            if cand_overlap < winner_overlap:
                winner = cand
                winner_overlap = cand_overlap
            i += 1
        # print(f"\tother: {other.id}")
        # print(f"\twinner: {winner.id}")
        # print(f"\toverlap: {winner_overlap}")
        return winner

    def tournamentSelectionUniqueGenealogyB(
        self,
        input_population: list[Individual],
        other: Individual = None,
        size: int = 1,
    ) -> Individual:
        """
        Implements a tournament selection, where it selects the best candidate among 'size' randomly selected ones from the 'input_population'.
        The winner is the one that has the least overlap of its 'linearized_genealogy' with 'linearized_genealogy' of the 'other' Individual.
        The 'linearized_genealogy' is a set of '<int1>/<int2>' without replacement extracted from the individual's 'genealogy' dictionary.
        NOTE: Here, the 'linearized_genealogy' is a set of the '<int1>' elements of '<int1>/<int2>' WITHOUT replacement extracted from the individual's 'genealogy' dictionary.
        """

        def linearized_genealogy(ind: Individual) -> set:
            # --- Collect the <int1> (id) part only, deduplicated into a set
            return {
                int(entry.split("/")[0])
                for entries in ind.genealogy.values()
                for entry in entries
            }

        other_lin = linearized_genealogy(other) if other is not None else set()

        # Dataset-aware: sample from currents only when any exist.
        input_population = self._tournament_pool(input_population)
        i = 1
        # print(f"\ntournamentSelectionUniqueGenealogyB")
        winner = input_population[SRConfig.r.integers(0, len(input_population))]
        winner_overlap = len(linearized_genealogy(winner) & other_lin)
        # print(f"\tcand_overlap: {winner_overlap}")
        while i < size:
            cand = input_population[SRConfig.r.integers(0, len(input_population))]
            cand_overlap = len(linearized_genealogy(cand) & other_lin)
            # print(f"\tcand_overlap: {cand_overlap}")
            if cand_overlap < winner_overlap:
                winner = cand
                winner_overlap = cand_overlap
            i += 1
        # print(f"\tother: {other.id}")
        # print(f"\twinner: {winner.id}")
        # print(f"\toverlap: {winner_overlap}")
        return winner

    def tournamentSelectionLeastUsed(
        self,
        input_population: list[Individual],
        other: Individual = None,
        size: int = 1,
    ) -> Individual:
        """
        Implements a tournament selection, where it selects the best candidate among 'size' randomly selected candidates from the 'input_population'.
        The winner is the candidate that has the smallest total number of occurences in a 'linearized_genealogy' of all Individuals in the 'input_population'.
        So, the total number of occurences is a sum of the candidate's occurences over all individuals in the input_population.
        By the occurence of a candidate individual in the 'linearized_genealogy' we mean that candidate.id==<int1> in the pair of '<int1>/<int2>'.
        NOTE: Here, the 'linearized_genealogy' is a list of '<int1>/<int2>' WITH replacement extracted from the individual's 'genealogy' dictionary.
        """
        # --- Build a usage-count lookup: how many times each id appears as <int1>
        # --- across the linearized genealogies (WITH replacement) of all individuals
        # --- in input_population.
        # print(f"\ntournamentSelectionLeastUsed")
        # Dataset-aware: sample from currents only when any exist. NOTE this
        # also restricts the usage-count denominator below to the chosen pool,
        # which is what we want - usage stats should reflect the candidate
        # set the tournament actually draws from.
        input_population = self._tournament_pool(input_population)
        usage: dict[int, int] = {}
        for ind in input_population:
            for entries in ind.genealogy.values():
                for entry in entries:  # entry = "<int1>/<int2>"
                    ind_id = int(entry.split("/")[0])
                    usage[ind_id] = usage.get(ind_id, 0) + 1

        i = 1
        winner = input_population[SRConfig.r.integers(0, len(input_population))]
        winner_count = usage.get(winner.id, 0)
        # print(f"\tcand_usage: {winner_count}")
        while i < size:
            cand = input_population[SRConfig.r.integers(0, len(input_population))]
            cand_count = usage.get(cand.id, 0)
            # print(f"\tcand_usage: {cand_count}")
            if cand_count < winner_count:
                winner = cand
                winner_count = cand_count
            i += 1
        # print(f"\twinner: {winner.id}")
        # print(f"\tusage: {winner_count}")
        return winner

    def reload_data(self):
        """
        Reloads data and waits until the file's mtime is stable across the read
        and a subsequent verification (i.e. no writer touched it during the read).
        Returns the verified timestamp.

        Aborts after MAX_ATTEMPTS rather than spinning forever if some external
        process keeps modifying the file. The previous implementation used
        `while True` with math.isclose(..., rel_tol=1e-9, abs_tol=1e-9) which, at
        Unix-timestamp magnitudes (~1.78e9), had an effective tolerance of ~1.78 s
        - large enough to mask actual file changes. We now use exact == on mtimes
        (both sides come straight from os.path.getmtime, so no float arithmetic
        is involved) and add a bounded retry budget with a short sleep.
        """
        MAX_ATTEMPTS = 20      # 20 * 0.1 s = up to ~2 s of waiting for the writer
        RETRY_SLEEP_S = 0.1

        curr_timestamp = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            _, curr_timestamp = initialize_config_and_topologies(
                self.cli_args,
                seed=SRConfig.seed,
                readTopology=False,
                start_time=self.start_time,
            )
            self.data_timestamp = curr_timestamp
            if curr_timestamp == os.path.getmtime(SRConfig.train_data):
                log(
                    f"Training data reloaded (attempt {attempt}/{MAX_ATTEMPTS})",
                    start_time=self.start_time,
                    file_path=SRConfig.log_file,
                    level="HIGHLIGHT_INFO",
                )
                self._advance_dataset_version()
                return curr_timestamp
            time.sleep(RETRY_SLEEP_S)

        # Did not converge: the file is still being written. Accept the last read
        # and warn so it shows up in the log.
        log(
            f"Training data did not stabilize after {MAX_ATTEMPTS} attempts; "
            f"accepting last read (read mtime={curr_timestamp}, "
            f"current mtime={os.path.getmtime(SRConfig.train_data)})",
            start_time=self.start_time,
            file_path=SRConfig.log_file,
            level="WARNING",
        )
        self._advance_dataset_version()
        return curr_timestamp

    def _advance_dataset_version(self):
        """Move the current_best/ mirror to the next dataset version,
        reset the archive, and zero per-individual backprop counters.

        Called from reload_data() after a (successful or fallback) reload.
        Sequence (order is significant):

          1. Flush any deferred dumps so the OUTGOING version's folder
             reflects the final state of the population under the old data.
          2. Bump self._dataset_version so all subsequent dumps land in the
             new current_best_<version>/ folder.
          3. Empty self.archive: best-so-far entries were measured on the
             old dataset and are not comparable to evaluations against the
             new one, so the pareto front is reset. (Done unconditionally,
             regardless of whether the on-disk mirror is enabled - the
             archive reset is part of the semantic data-change response,
             not just a mirror concern.)
          4. Zero backprop_iters for every individual in mainPopulation and
             offspring. The counter measures backprop work spent against the
             current dataset; once the dataset changes, the prior count is
             no longer comparable across the boundary (e.g. for maturity-
             based comparisons that read backprop_iters as "training effort
             on the data we're evaluating on"). Done unconditionally.
          5. Materialize the new folder immediately with an unthrottled
             dump, so it appears as soon as the reload completes (and so
             the new archive folder records the post-clear empty state).

        Note: the version counter advances even when the mirror is disabled
        (gate False or SRConfig.saveCurrentBest off), so dumps stay in sync
        if the flag is re-enabled mid-run.
        """
        # 1. Flush any deferred dumps to the OLD folder before advancing.
        self.flush_pending_dumps()

        # 2. Bump version - all subsequent dumps go into the new folder.
        self._dataset_version += 1
        log(
            f"current_best mirror advanced to dataset version "
            f"{self._dataset_version}",
            start_time=self.start_time,
            file_path=SRConfig.log_file,
            level="HIGHLIGHT_INFO",
        )

        # 3. Reset the archive. clear() goes through the _PopulationListView
        # wrapper, which fires the throttled archive-dump callback - that
        # would set _dirty_archive and defer the write. The unconditional
        # _do_dump_current_best_archive() call below discharges it anyway,
        # so the final state is consistent regardless of throttling.
        archive_size_before = len(self.archive)
        self.archive.clear()
        log(
            f"Archive cleared after data reload "
            f"(removed {archive_size_before} individuals); dataset version "
            f"is now {self._dataset_version}",
            start_time=self.start_time,
            file_path=SRConfig.log_file,
            level="HIGHLIGHT_INFO",
        )

        # 4. Per-individual reset for every member of mainPopulation and
        # offspring. Three concerns folded into one loop:
        #
        #   - backprop_iters and age_last_change_backprop: both count work
        #     spent against the current dataset and become incomparable
        #     across a data change.
        #   - adult flag: re-derived from the configured age field so the
        #     maturity flag stays consistent with its inputs (e.g. if
        #     SRConfig.ageToConsider == "backprop_iters" or
        #     "age_last_change_backprop", a freshly-reset individual is no
        #     longer adult; under "age" it remains adult since age itself
        #     is not reset).
        #   - perf_history: wiped to {} so no entry measured against the
        #     old dataset can ever be returned by latest_perf() or
        #     accessed by code that walks the dict. last_perf is left in
        #     place (still stale) so the four bypass sites we hardened
        #     (sortByXxx, splitPopulationByDomination, log_population_state,
        #     tournamentSelectionRMSE) can degrade gracefully via
        #     latest_perf()'s fallback to last_perf. The dataset-aware
        #     dominance helper still flags last_perf as stale so it never
        #     wins a comparison against a freshly-measured current peer.
        #
        # Modifying the Individual objects in place also covers references
        # held by self.queue, self.waitingNotTuned, self.currentlyTuned,
        # self.mainPopulationNotTuned, etc. - those collections hold the
        # same object identities, not copies.
        reset_count = 0
        for ind in list(self.mainPopulation) + list(self.offspring):
            ind.backprop_iters = 0
            ind.age_last_change_backprop = 0
            # backprop_iters reset to 0 -> the pruning watermark must reset too,
            # otherwise (backprop_iters - lastPruningBackprops) would go negative.
            ind.lastPruningBackprops = 0
            ind.adult = (
                getattr(ind, SRConfig.ageToConsider) >= SRConfig.th_adult
            )
            ind.perf_history = {}
            reset_count += 1
        log(
            f"Reset backprop_iters, age_last_change_backprop, and perf_history "
            f"on {reset_count} individuals (mainPopulation + offspring) after "
            f"data reload; per-individual training effort and per-age "
            f"performance snapshots are now counted against dataset "
            f"version {self._dataset_version}",
            start_time=self.start_time,
            file_path=SRConfig.log_file,
            level="HIGHLIGHT_INFO",
        )

        # 5. Materialize the new folder immediately, bypassing the throttle
        # so the user sees current_best_<new>/ appear as soon as the reload
        # completes (instead of having to wait for the next mutation). The
        # archive folder created here reflects the just-emptied archive.
        if getattr(self, "_dump_enabled", False) and getattr(
            SRConfig, "saveCurrentBest", True
        ):
            self._do_dump_current_best_main_population()
            self._do_dump_current_best_archive()

    def _apply_mutation(self, sub_topology, parent_id):
        """
        Applies either activity mutation or config mutation to the given sub-topology.
        """
        child = sub_topology.cloneSubTopology()
        if SRConfig.r.random() < SRConfig.applyActivityOrConfigMutation:
            getattr(self.masterTopology, SRConfig.networkActivityMutation)(child)
        else:
            self.masterTopology.networkConfigMutation_B(child)
        return child

    def _apply_reproduction(self, sub_topology, parent_id):
        """
        Replicates the given sub-topology.
        """
        child = sub_topology.cloneSubTopology()
        log_message = "Created new individual using reproduction"
        if parent_id is not None:
            log_message += f" of parent {parent_id}"
        log(log_message + ".", start_time=self.start_time, file_path=SRConfig.log_file)
        # # --- TODO: Remove the check belwo; it is just for debugging purposes
        # if not SubTopology.compareSubtopologies(
        #     subtopology1=child, subtopology2=sub_topology
        # ):
        #     log(
        #         "The new and original subtopologies are NOT identical.",
        #         start_time=self.start_time,
        #         file_path=SRConfig.log_file,
        #     )
        # # ---
        return child

    def crossover_mutation_reproduction(self, regularize: bool, triggerId: int = -1):
        """
        Generates a new individual using the following steps:
          1) Generate a set of candidate subtopologies, C, of size 'SRConfig.generated_set_size'
          2) Tune weights of each subtopology in C using 'SRConfig.nbOfNewbornIterations' backprop iterations
          3) Create an Individual from the chosen solution
          4) Find a set, N, of non-dominated solutions in C
          5) Choose randomly one solution from set N
          6) Return a list of new Individuals
        """
        log(
            f"Started crossover_mutation_reproduction() triggered by individual {triggerId}",
            self.start_time,
            file_path=SRConfig.log_file,
            level="DEBUG",
        )
        # --- TODO: check, if this is always needed
        sorted_main_population = nonDominatedSorting(
            population=copy.deepcopy(self.mainPopulation),
            finalSize=len(self.mainPopulation),
            criteria=SubTopology.keysAll,
        )
        # sorted_main_population = copy.deepcopy(self.mainPopulation)

        # --- Update node utilizations
        self.masterTopology.resetNodeUtilizations()
        for ind in self.mainPopulation:
            self.masterTopology.updateNodeUtilizations(
                ind.last_perf.activeNodesCoordinates
            )

        # --- 1) Generate a set of candidate subtopologies, C
        log(
            f"Started generating candidates triggered by individual {triggerId}",
            self.start_time,
            file_path=SRConfig.log_file,
            level="DEBUG",
        )
        candidates = []
        newOffspring: Individual = None
        reproduction = False
        for i in range(SRConfig.generated_set_size):
            # --- Select new parents for each crossover/mutation action
            while True:
                p1: Individual = getattr(self, SRConfig.selection_1)(
                    input_population=sorted_main_population,
                    other=None,
                    size=SRConfig.tournamentSize1,
                )
                p1_last_perf = p1.last_perf
                if p1_last_perf.performance["nbOfActiveNodes"] < SRConfig.maxFrontNb:
                    break
            while True:
                p2: Individual = getattr(self, SRConfig.selection_2)(
                    input_population=sorted_main_population,
                    other=p1,
                    size=SRConfig.tournamentSize2,
                )
                p2_last_perf = p2.last_perf
                if (
                    p2.id != p1.id
                    and p2_last_perf.performance["nbOfActiveNodes"]
                    < SRConfig.maxFrontNb
                ):
                    break

            # --- Create subtopologies from parents
            log(
                f"Started creating subtopologies from parents triggered by individual {triggerId}",
                self.start_time,
                file_path=SRConfig.log_file,
                level="DEBUG",
            )
            p1_SubTopology = SubTopology.createSubtopologyFromParameters(
                activeNodesCoordinates=p1_last_perf.activeNodesCoordinates,
                nnWeights=p1_last_perf.nn_weights,
                losses=p1_last_perf.performance,
                text="crossover_mutation_reproduction_p1",
            )
            p2_SubTopology = SubTopology.createSubtopologyFromParameters(
                activeNodesCoordinates=p2_last_perf.activeNodesCoordinates,
                nnWeights=p2_last_perf.nn_weights,
                losses=p2_last_perf.performance,
                text="crossover_mutation_reproduction_p2",
            )

            # --- Try to generate offspring subtopology no larger than its parents
            nbOfOffspringGenerationAttempts = 1
            while nbOfOffspringGenerationAttempts > 0:
                first_parent_only = False
                # --- Generate a new subtopology using reproduction
                if SRConfig.crossoverRate == 0.0:
                    cand = self._apply_reproduction(p1_SubTopology, p1.id)
                    age = p1.age
                    reproduction = True
                    first_parent_only = True

                # --- Generate a new subtopology using crossover and mutation
                elif SRConfig.r.random() < SRConfig.crossoverRate:
                    # cand, _ = self.masterTopology.subTopology_crossover_limited_arity(p1_SubTopology, p2_SubTopology)
                    cand, _ = getattr(self.masterTopology, SRConfig.crossover)(
                        p1_SubTopology, p2_SubTopology
                    )
                    age = min(p1.age, p2.age)
                    if SRConfig.r.random() < SRConfig.mutationAfterCrossoverRate:
                        cand = self._apply_mutation(cand, None)

                # --- Generate a new subtopology using only mutation
                else:
                    cand = self._apply_mutation(p1_SubTopology, p1.id)
                    age = p1.age
                    first_parent_only = True

                self.masterTopology.rectifyActiveNodesConnectivity(cand)
                nbOfOffspringGenerationAttempts -= 1

                # --- Check the size of the 'cand' subtopology
                if (
                    first_parent_only
                    and cand.nbOfActiveNodes <= p1_SubTopology.nbOfActiveNodes
                ):
                    break
                if (not first_parent_only) and cand.nbOfActiveNodes <= max(
                    p1_SubTopology.nbOfActiveNodes, p2_SubTopology.nbOfActiveNodes
                ):
                    break

            # --- Load new data
            #     Ensures the offspring generated below get tuned against the latest
            #     training data. Exact != comparison: see the note in
            #     _handle_not_subject_to_tune - math.isclose(..., rel_tol=1e-9) had an
            #     effective tolerance of ~1.78 s at Unix-timestamp magnitudes, so it
            #     silently missed sub-second file changes.
            if self.data_timestamp != os.path.getmtime(SRConfig.train_data):
                self.reload_data()

            # --- 2) Tune weights of each newly generated candidate subtopology
            # nbOfTuningIters = SRConfig.nbOfNewbornIterations if not SRConfig.standard_EA else 0
            log(
                f"Started tuning weights of newly generated subtopology triggered by individual {triggerId}",
                self.start_time,
                file_path=SRConfig.log_file,
                level="DEBUG",
            )
            nbOfTuningIters = 0
            try:
                if SRConfig.nbOfNewbornIterations < 0:
                    # --- clip weights and update losses
                    cand.updateLearnableParamsMaskAndValues()  # clip weights
                    cand.updateActiveNodes()
                    cand.losses["valid_loss"] = cand.calculateValidationLoss()
                    c_raw, c_weighted = cand.calculateConstrainViolationsRMSE()
                    cand.losses["rmse_constr"] = c_raw
                else:
                    _, nbOfImprovements = cand.train_nn(
                        learningSteps=nbOfTuningIters,
                        learningRate=SRConfig.learning_rate_newborn,
                        regularize=True,
                        constrain=True,
                        clipWeights=True,
                        backpropItersBefore=0,
                        lastPruningBackprops=0,
                    )
                self.totalBackpropsUsed += nbOfTuningIters
            except Exception as e:
                log(
                    f"Exception in cand.train_nn: {e}",
                    start_time=self.start_time,
                    file_path=SRConfig.log_file,
                    level="HIGHLIGHT_RED",
                )
                continue

            # --- Check if the performance of the candidate is different from the parents
            isClosePrecision = 1e-3
            if not reproduction and (
                (
                    math.isclose(
                        cand.losses["valid_loss"],
                        p1.last_perf.performance["valid_loss"],
                        rel_tol=isClosePrecision,
                    )
                    and math.isclose(
                        cand.losses["rmse_constr"],
                        p1.last_perf.performance["rmse_constr"],
                        rel_tol=isClosePrecision,
                    )
                )
                or (
                    (not first_parent_only)
                    and math.isclose(
                        cand.losses["valid_loss"],
                        p2.last_perf.performance["valid_loss"],
                        rel_tol=isClosePrecision,
                    )
                    and math.isclose(
                        cand.losses["rmse_constr"],
                        p2.last_perf.performance["rmse_constr"],
                        rel_tol=isClosePrecision,
                    )
                )
            ):
                log(
                    f"Performance of the candidate created by crossover is very close (rel_tol={isClosePrecision}) to performance of some of its parents.",
                    start_time=self.start_time,
                    file_path=SRConfig.log_file,
                    level="DEBUG",
                )
                continue

            # --- 3. Create new individual from 'cand' subTopology
            log(
                f"Started creating an individual from new subtopology triggered by individual {triggerId}",
                self.start_time,
                file_path=SRConfig.log_file,
                level="DEBUG",
            )
            if cand.params_2_learn and cand.checkUnitIdent1Nodes():
                params = cand.getSubtopologyParameters()
                performance = {
                    "valid_loss": cand.losses["valid_loss"],
                    "rmse_constr": cand.losses["rmse_constr"],
                    "nbOfActiveNodes": cand.losses["nbOfActiveNodes"],
                    "complexity": cand.losses["complexity"],
                    "nbOfImprovements": 1,  # --- Any positive value
                }

                history_update = PerformanceHistory(
                    performance=performance,
                    nn_weights=params["nnParams"],
                    activeNodesCoordinates=params["activeNodesCoordinates"],
                    dataTimestamp=self.data_timestamp,
                )
                log(
                    f"PerformanceHistory created for a new individual triggered by individual {triggerId}",
                    self.start_time,
                    file_path=SRConfig.log_file,
                    level="DEBUG",
                )

                # --- Create 'new_genealogy' dict by merging (p1.genealogy, p2.genealogy) such that
                # --- tuples with the same key are merged without replacement in a new tuple.
                # --- Example: p1.genealogy = {0: (), 1: (12, 5), 2: (23, 16)}
                # ---          p2.genealogy = {0: (), 1: (7, 12)}
                # ---          new_genealogy = {0: (), 1: (5, 7, 12), 2: (23, 16)}
                all_keys = (
                    p1.genealogy.keys()
                    if first_parent_only
                    else p1.genealogy.keys() | p2.genealogy.keys()
                )
                new_genealogy = {
                    k: tuple(
                        sorted(
                            set(p1.genealogy.get(k, ())) | set(p2.genealogy.get(k, ()))
                        )
                    )
                    for k in all_keys
                }
                parents = (f"{p1.id}/{p1.backprop_iters}",)
                if not first_parent_only:
                    parents = (
                        f"{p1.id}/{p1.backprop_iters}",
                        f"{p2.id}/{p2.backprop_iters}",
                    )
                individual = Individual(
                    inPopulation=None,
                    indId=p1.id if reproduction else i,
                    iters=age,
                    parents=parents,
                    genealogy=new_genealogy,
                )

                individual.backprop_iters = nbOfTuningIters
                # --- Carry the newborn's pruning watermark from its initial tuning
                #     (cand.train_nn may have pruned if nbOfTuningIters >= pruningStep).
                individual.lastPruningBackprops = getattr(
                    cand, "lastPruningBackprops", 0
                )
                individual.adult = (
                    True if individual.backprop_iters >= SRConfig.th_adult else False
                )
                individual.perf_history[getattr(individual, SRConfig.ageToConsider)] = (
                    history_update
                )
                individual.last_perf = history_update
                individual.last_output = (
                    cand.get_subtopology_output(SRData.x_data).detach().cpu().numpy()
                )
                log(
                    f"Generated output of a new individual triggered by individual {triggerId}",
                    self.start_time,
                    file_path=SRConfig.log_file,
                    level="DEBUG",
                )
                # --- Generate expression and simple expression
                cand.getAnalyticFormula(simplify=False)
                log(
                    f"Analytic formula generated for a new individual triggered by individual {triggerId}",
                    self.start_time,
                    file_path=SRConfig.log_file,
                    level="DEBUG",
                )
                outputNode: NNUnit = cand.nnLayers[-1].units[0]
                individual.expr = outputNode.string_analytic_expression
                # --- TODO: always set to "NA"
                # if outputNode.symbolic_expression:
                #     individual.simpleExpr = outputNode.symbolic_expression.simpleExpr
                # else:
                #     individual.simpleExpr = "NA"
                individual.simpleExpr = "NA"

                candidates.append(individual)
            else:
                if not reproduction:
                    log(
                        f"Child created by crossover/mutation triggered by individual {triggerId} is not valid.",
                        start_time=self.start_time,
                        file_path=SRConfig.log_file,
                        level="DEBUG",
                    )
                else:
                    log(
                        f"Child created by reproduction triggered by individual {triggerId} is not valid.",
                        start_time=self.start_time,
                        file_path=SRConfig.log_file,
                        level="DEBUG",
                    )

        if candidates:
            # --- 4) Find a set, N, of non-dominated solutions in 'candidates'
            # if not reproduction:
            #     log_population_state(
            #         population=candidates,
            #         start_time=self.start_time,
            #         file_path=SRConfig.log_file,
            #         title=f"Candidates created using crossover/mutation triggered by individual {triggerId}",
            #     )
            # else:
            #     log_population_state(
            #         population=candidates,
            #         start_time=self.start_time,
            #         file_path=SRConfig.log_file,
            #         title=f"Candidates created by reproduction triggered by individual {triggerId}",
            #     )
            nonDominatedSet, _ = splitPopulationByDomination(
                candidates, front=0, criteria=SubTopology.keysAll
            )
            # log_population_state(
            #     population=nonDominatedSet,
            #     start_time=self.start_time,
            #     file_path=SRConfig.log_file,
            #     title=f"Non-dominated individuals of 'candidates' triggered by individual {triggerId}",
            # )

            if SRConfig.standard_EA:
                # # ---  5a) Choose randomly one solution from set N
                # newOffspring = [SRConfig.r.choice(nonDominatedSet)]
                # newOffspring[0].id = self.idCounter
                # self.idCounter += 1
                # ---  5b) Take all solutions from nonDominatedSet
                newOffspring = nonDominatedSet
                for ind in newOffspring:
                    ind.id = self.idCounter
                    self.idCounter += 1
            else:
                # ---  5b) Take all solutions from nonDominatedSet
                newOffspring = nonDominatedSet
                for ind in newOffspring:
                    ind.id = self.idCounter
                    self.idCounter += 1

            # --- Perturbation: optional
            if (SRConfig.perturbationIntervalIndividuals > 0) and (
                self.idCounter
                > self.lastPerturbationIdCounter
                + SRConfig.perturbationIntervalIndividuals
            ):
                self.lastPerturbationIdCounter = self.idCounter
                self.perturbPopulation()
                return None
            # ---
            elif (SRConfig.perturbationIntervalBackprops > 0) and (
                self.totalBackpropsUsed
                > self.lastPerturbationIdBackprops
                + SRConfig.perturbationIntervalBackprops
            ):
                self.lastPerturbationIdBackprops = self.totalBackpropsUsed
                self.perturbPopulation()
                return None

        return newOffspring

    def initRandomPopulation(
        self,
        masterTopology: MasterTopology,
        initPopulationSize: int,
    ):
        """
        Initializes the population at the very beginning.
        No learning is applied to the generated subtopologies.
        """

        population = []
        for k in range(initPopulationSize):
            # subTopology = masterTopology.generateRandomSubTopology()
            subTopology = masterTopology.generate_random_subtopology_limited_arity()
            _, nbOfImprovements = subTopology.train_nn(
                learningSteps=0,
                learningRate=SRConfig.learning_rate_newborn,
                regularize=False,
                clipWeights=True,
            )
            individual = Individual(inPopulation="mainPopulation", indId=self.idCounter)
            performance = {
                "valid_loss": subTopology.losses["valid_loss"],
                "rmse_constr": subTopology.losses["rmse_constr"],
                "nbOfActiveNodes": subTopology.losses["nbOfActiveNodes"],
                "complexity": subTopology.losses["complexity"],
                "nbOfImprovements": nbOfImprovements,
            }
            # --- filePath=SRConfig.outputNamePrefix + f"subTopology_definition_{k}.txt"
            params = subTopology.getSubtopologyParameters()
            individual.perf_history[getattr(individual, SRConfig.ageToConsider)] = (
                PerformanceHistory(
                    performance=performance,
                    nn_weights=params["nnParams"],
                    activeNodesCoordinates=params["activeNodesCoordinates"],
                    dataTimestamp=self.data_timestamp,
                )
            )
            individual.last_perf = individual.perf_history[
                getattr(individual, SRConfig.ageToConsider)
            ]
            population.append(individual)
            self.mainPopulationProcessed[individual.id] = 0
            self.idCounter += 1

        # --- Report populations sorted by complexity
        population.sort(key=lambda ind: ind.last_perf.performance["complexity"])
        log_population_state(
            population=population,
            start_time=self.start_time,
            title="Initialization (no learning): Individuals in main population",
        )
        return population

    def initPopulationFromDirectory(self, path: str):
        """
        Inits new population from files in the given directory.
        The format of files it predefined - it stores on which position each node is active and what are the weights of the neural network.
        """
        population = []
        files = []
        for fn in os.listdir(path):
            fp = os.path.join(path, fn)
            if os.path.isfile(fp):
                files.append(fp)

        if not files:
            log(
                f"No files found in directory {path}.",
                start_time=self.start_time,
                file_path=SRConfig.log_file,
            )
            return population

        for file_path in files:
            try:
                ind = Individual.loadIndividualFromFile(
                    filePath=file_path, ind_id=self.idCounter
                )
            except Exception as e:
                log(
                    f"[initPopulationFromDirectory] Skipping '{file_path}': {e}",
                    start_time=self.start_time,
                    file_path=SRConfig.log_file,
                )
                continue

            ind.inPopulation = "mainPopulation"
            self.mainPopulationProcessed[ind.id] = 0
            population.append(ind)
            self.idCounter += 1

        # --- Report population sorted by complexity
        population.sort(key=lambda ind: ind.last_perf.performance["complexity"])
        log_population_state(
            population=population,
            start_time=self.start_time,
            title=f"Initialization from directory: Individuals in main population ({path})",
        )
        return population

    def initPopulationFromExpressions(self, path):
        """
        Inits new population from a file with one expression per line.
        If less expressions than the required population size are found, the rest of the population is initialized randomly.
        """
        population = []
        try:
            with open(path, "r") as file:
                lines = file.readlines()
                for line in lines:
                    expr = line.strip()
                    if not expr or expr[0] == "#":
                        continue
                    expr = re.sub(
                        r"(?<!^)(?<!\+)-", "+-", expr
                    )  # replace '-' with '+-' except at the start
                    try:
                        if expr:
                            # TODO: handle incorrect inputs
                            pref_expr = infix_to_prefix(
                                expr, self.masterTopology.availableOperators
                            )
                            subTopology = self.masterTopology.ExprToSubTopology(
                                pref_expr
                            )
                            if not subTopology:
                                log(
                                    f"Could not create subtopology from expression '{expr}'.",
                                    start_time=self.start_time,
                                    file_path=SRConfig.log_file,
                                )
                                continue

                            _, nbOfImprovements = subTopology.train_nn(
                                learningSteps=0,
                                learningRate=SRConfig.learning_rate_newborn,
                                regularize=False,
                                clipWeights=True,
                            )

                            individual = Individual(
                                inPopulation="mainPopulation", indId=self.idCounter
                            )

                            individual.expr = subTopology.getAnalyticFormula()

                            performance = {
                                "valid_loss": subTopology.losses["valid_loss"],
                                "rmse_constr": subTopology.losses["rmse_constr"],
                                "nbOfActiveNodes": subTopology.losses[
                                    "nbOfActiveNodes"
                                ],
                                "complexity": subTopology.losses["complexity"],
                                "nbOfImprovements": nbOfImprovements,
                            }

                            params = subTopology.getSubtopologyParameters()
                            individual.perf_history[
                                getattr(individual, SRConfig.ageToConsider)
                            ] = PerformanceHistory(
                                performance=performance,
                                nn_weights=params["nnParams"],
                                activeNodesCoordinates=params["activeNodesCoordinates"],
                                dataTimestamp=self.data_timestamp,
                            )
                            individual.last_perf = individual.perf_history[
                                getattr(individual, SRConfig.ageToConsider)
                            ]
                            population.append(individual)
                            if len(population) == SRConfig.init_pop_size:
                                break
                            self.mainPopulationProcessed[individual.id] = 0
                            self.idCounter += 1
                    except Exception as e:
                        log(
                            f"Error processing expression '{expr}': {e}",
                            start_time=self.start_time,
                            file_path=SRConfig.log_file,
                            level="ERROR",
                        )
                        continue

        except FileNotFoundError:
            log(
                f"File not found: {path}",
                start_time=self.start_time,
                file_path=SRConfig.log_file,
                level="ERROR",
            )
            return population
        except Exception as e:
            log(
                f"Error reading file '{path}': {e}",
                start_time=self.start_time,
                file_path=SRConfig.log_file,
                level="ERROR",
            )
            return population

        # --- If the file does not contain enough expressions, initialize the rest randomly
        if len(population) < SRConfig.init_pop_size:
            remaining_size = SRConfig.init_pop_size - len(population)
            random_population = self.initRandomPopulation(
                self.masterTopology, initPopulationSize=remaining_size
            )
            population.extend(random_population)

        # --- Report population sorted by complexity
        population.sort(key=lambda ind: ind.last_perf.performance["complexity"])
        log_population_state(
            population=population,
            start_time=self.start_time,
            file_path=SRConfig.log_file,
            title=f"Initialization from file: Individuals in main population ({path})",
        )
        return population

    def initialPopulationLearningPhaseI(self):
        """
        Applies learning to the individuals in initial main population.
        This is done only once after the initial main population is created.
        Phase-I:
           reg2rmseMainRun: float = 0.0
           constr2rmse: float = 0.0
           sng2rmse: float = 0.0
        """
        # ------------------------------------------------
        # --- Dask initialization
        # ------------------------------------------------
        n_cores = 4  # os.cpu_count()
        initial = deque(islice(self.mainPopulation, n_cores))
        queue = deque(islice(self.mainPopulation, n_cores, None))
        cluster = LocalCluster(
            n_workers=n_cores,
            threads_per_worker=1,
            processes=True,
            # env={"DASK_DISTRIBUTED__ADMIN__GC_DIAGNOSTICS": "false"},
        )
        client = Client(cluster)

        # ------------------------------------------------
        # --- Main body of the population tuning
        # ------------------------------------------------
        # --- Parameters specific to initialPopulationLearning
        extra_args = (
            "--learning_rate",
            f"{SRConfig.learning_rate_newborn}",
            "--iters",
            f"{int(SRConfig.totalItersInit / 2)}",
            "--reg2rmse",
            "0.0",
            "--constr2rmse",
            "0.0",
            "--sng2rmse",
            "0.0",
        )

        futures = self.submit_initial_tasks(client, initial, specific_params=extra_args)
        future_to_key = {fut: ind_id for ind_id, fut in futures.items()}
        ac = as_completed(futures.values())
        population = []
        for future in ac:
            key = future_to_key.pop(future, None)
            if key is not None:
                futures.pop(key, None)
            try:
                result = future.result()
                if result is None:
                    future.release()
                    continue
            except Exception as e:
                log(
                    f"Exception in worker: {e}",
                    start_time=self.start_time,
                    file_path=SRConfig.log_file,
                )
                future.release()
                continue

            for text in result["logs"]:
                log(f"{text}", start_time=self.start_time, file_path=SRConfig.log_file)

            individual, readyNotTuned = self.process_individual(result)
            population.append(individual)

            future.release()

            # --- Pick individual from queue
            while queue:
                log(
                    f"Picked up next individual of ({queue[0].inPopulation}) from queue with id {queue[0].id}.",
                    start_time=self.start_time,
                    file_path=SRConfig.log_file,
                    level="DEBUG",
                )
                self.submit_next_task(
                    client,
                    queue,
                    futures,
                    ac,
                    future_to_key,
                    specific_params=extra_args,
                )
                break
        self.mainPopulation = population

        # --- Report populations sorted by complexity
        self.mainPopulation.sort(
            key=lambda ind: ind.last_perf.performance["complexity"]
        )
        log_population_state(
            population=self.mainPopulation,
            start_time=self.start_time,
            title="Initialization (after phase-I): Individuals in main population",
            file_path=SRConfig.outputNamePrefix + "initial_population_phase_I.txt",
        )

        # --- 1. Cancel all futures if hard finish is triggered
        try:
            client.cancel(
                futures, force=True
            )  # --- force=True is important for stubborn tasks
            wait(
                futures, timeout="30s"
            )  # --- ensure scheduler processes the cancellation
            log(
                f"All futures canceled: initialPopulationLearningPhaseI.",
                start_time=self.start_time,
                file_path=SRConfig.log_file,
                level="DEBUG",
            )
        except Exception:
            log(
                f"Exception while canceling futures: initialPopulationLearningPhaseI.",
                start_time=self.start_time,
                file_path=SRConfig.log_file,
                level="DEBUG",
            )
            pass
        try:
            client.close()
        except Exception:
            log(
                f"Exception while closing client: initialPopulationLearningPhaseI.",
                start_time=self.start_time,
                file_path=SRConfig.log_file,
            )
        finally:
            cluster.close(timeout="30s")  # --- give workers time to die

    def learningPhaseI_without_constraints(self, population: list[Individual]):
        """
        Population is tuned using
           - reg2rmseMainRun: float = 0.0
           - constr2rmse: float = 0.0
           - sng2rmse: float = 0.0
        """
        # ------------------------------------------------
        # --- Dask initialization
        # ------------------------------------------------
        n_cores = 4  # os.cpu_count()
        initial = deque(islice(population, n_cores))
        queue = deque(islice(population, n_cores, None))
        cluster = LocalCluster(
            n_workers=n_cores,
            threads_per_worker=1,
            processes=True,
        )
        client = Client(cluster)

        # ------------------------------------------------
        # --- Main body of the population tuning
        # ------------------------------------------------
        # --- Parameters specific to initialPopulationLearning
        extra_args = (
            "--learning_rate",
            f"{SRConfig.learning_rate_newborn}",
            "--iters",
            f"{int(SRConfig.totalItersInit / 2)}",
            "--reg2rmse",
            "0.0",
            "--constr2rmse",
            "0.0",
            "--sng2rmse",
            "0.0",
        )

        futures = self.submit_initial_tasks(client, initial, specific_params=extra_args)
        future_to_key = {fut: ind_id for ind_id, fut in futures.items()}
        ac = as_completed(futures.values())
        res_population = []
        for future in ac:
            key = future_to_key.pop(future, None)
            if key is not None:
                futures.pop(key, None)
            try:
                result = future.result()
                if result is None:
                    future.release()
                    continue
            except Exception as e:
                future.release()
                continue

            for text in result["logs"]:
                log(f"{text}", start_time=self.start_time, file_path=SRConfig.log_file)

            individual, readyNotTuned = self.process_individual(result)
            res_population.append(individual)

            future.release()

            # --- Pick individual from queue
            while queue:
                log(
                    f"Picked up next individual of ({queue[0].inPopulation}) from queue with id {queue[0].id}.",
                    start_time=self.start_time,
                    file_path=SRConfig.log_file,
                    level="DEBUG",
                )
                self.submit_next_task(
                    client,
                    queue,
                    futures,
                    ac,
                    future_to_key,
                    specific_params=extra_args,
                )
                break

        # --- Report populations sorted by complexity
        res_population.sort(key=lambda ind: ind.last_perf.performance["complexity"])
        log_population_state(
            population=res_population,
            start_time=self.start_time,
            title="Initialization (after phase-I) without constraints: Individuals in main population",
            file_path=SRConfig.outputNamePrefix
            + "initial_population_phase_I_noconstraints.txt",
        )

        # --- 1. Cancel all futures if hard finish is triggered
        try:
            client.cancel(
                futures, force=True
            )  # --- force=True is important for stubborn tasks
            wait(
                futures, timeout="30s"
            )  # --- ensure scheduler processes the cancellation
            log(
                f"All futures canceled: initialPopulationLearningPhaseII.",
                start_time=self.start_time,
                file_path=SRConfig.log_file,
                level="DEBUG",
            )
        except Exception:
            log(
                f"Exception while canceling futures: initialPopulationLearningPhaseII.",
                start_time=self.start_time,
                file_path=SRConfig.log_file,
                level="DEBUG",
            )
            pass
        try:
            client.close()
        except Exception:
            log(
                f"Exception while closing client: initialPopulationLearningPhaseII.",
                start_time=self.start_time,
                file_path=SRConfig.log_file,
            )
        finally:
            cluster.close(timeout="30s")  # --- give workers time to die

        return res_population

    def learningPhaseI_with_constraints(self, population: list[Individual]):
        """
        Population is tuned using
           - reg2rmseMainRun: float = 0.0
           - constr2rmse: float = SRConfig.constr2rmseMainRun
           - sng2rmse: float = 0.0
        """
        # ------------------------------------------------
        # --- Dask initialization
        # ------------------------------------------------
        n_cores = 4  # os.cpu_count()
        initial = deque(islice(population, n_cores))
        queue = deque(islice(population, n_cores, None))
        cluster = LocalCluster(
            n_workers=n_cores,
            threads_per_worker=1,
            processes=True,
        )
        client = Client(cluster)

        # ------------------------------------------------
        # --- Main body of the population tuning
        # ------------------------------------------------
        # --- Parameters specific to initialPopulationLearning
        extra_args = (
            "--learning_rate",
            f"{SRConfig.learning_rate_newborn}",
            "--iters",
            f"{int(SRConfig.totalItersInit / 2)}",
            "--reg2rmse",
            "0.0",
            "--constr2rmse",
            f"{SRConfig.constr2rmseMainRun}",
            "--sng2rmse",
            "0.0",
        )

        futures = self.submit_initial_tasks(client, initial, specific_params=extra_args)
        future_to_key = {fut: ind_id for ind_id, fut in futures.items()}
        ac = as_completed(futures.values())
        res_population = []
        for future in ac:
            key = future_to_key.pop(future, None)
            if key is not None:
                futures.pop(key, None)

            try:
                result = future.result()
                if result is None:
                    future.release()
                    continue
            except Exception as e:
                future.release()
                continue

            for text in result["logs"]:
                log(f"{text}", start_time=self.start_time, file_path=SRConfig.log_file)

            individual, readyNotTuned = self.process_individual(result)
            res_population.append(individual)

            future.release()

            # --- Pick individual from queue
            while queue:
                log(
                    f"Picked up next individual of ({queue[0].inPopulation}) from queue with id {queue[0].id}.",
                    start_time=self.start_time,
                    file_path=SRConfig.log_file,
                    level="DEBUG",
                )
                self.submit_next_task(
                    client,
                    queue,
                    futures,
                    ac,
                    future_to_key,
                    specific_params=extra_args,
                )
                break

        # --- Report populations sorted by complexity
        res_population.sort(key=lambda ind: ind.last_perf.performance["complexity"])
        log_population_state(
            population=res_population,
            start_time=self.start_time,
            title="Initialization (after phase-I) with constraints: Individuals in main population",
            file_path=SRConfig.outputNamePrefix
            + "initial_population_phase_I_constraints.txt",
        )

        # --- 1. Cancel all futures if hard finish is triggered
        try:
            client.cancel(
                futures, force=True
            )  # --- force=True is important for stubborn tasks
            wait(
                futures, timeout="30s"
            )  # --- ensure scheduler processes the cancellation
            log(
                f"All futures canceled.",
                start_time=self.start_time,
                file_path=SRConfig.log_file,
                level="DEBUG",
            )
        except Exception:
            pass
        try:
            client.close()
        finally:
            cluster.close(timeout="30s")  # --- give workers time to die

        return res_population

    def initialPopulationLearningPhaseI_combined(self):
        """
        Applies learning to the individuals in initial main population.
        This is done only once after the initial main population is created.
        In Phase-I, half of the mainPopulation is tuned using
           - reg2rmseMainRun: float = 0.0
           - constr2rmse: float = 0.0
           - sng2rmse: float = 0.0
        and the other half is tuned using
           - reg2rmseMainRun: float = 0.0
           - constr2rmse: float = SRConfig.constr2rmseMainRun
           - sng2rmse: float = 0.0
        """
        population = []
        first_half = self.learningPhaseI_without_constraints(
            population=self.mainPopulation[: len(self.mainPopulation) // 2]
        )
        second_half = self.learningPhaseI_with_constraints(
            population=self.mainPopulation[len(self.mainPopulation) // 2 :]
        )
        self.mainPopulation = first_half + second_half

    def initialPopulationLearningPhaseII(self):
        """
        Applies learning to the individuals in initial main population.
        This is done only once after the initial main population is created.
        Phase-I if SRConstraints.constraintModules IS NOT empty:
           reg2rmseMainRun: float = 0.0
           constr2rmse: float = SRConfig.constr2rmseMainRun
           sng2rmse: float = 0.0
        Phase-I if SRConstraints.constraintModules IS empty:
           reg2rmseMainRun: float = SRConfig.reg2rmseMainRun
           constr2rmse: float = 0.0
           sng2rmse: float = 0.0
        """
        # ------------------------------------------------
        # --- Dask initialization
        # ------------------------------------------------
        n_cores = 4  # os.cpu_count()
        initial = deque(islice(self.mainPopulation, n_cores))
        queue = deque(islice(self.mainPopulation, n_cores, None))
        cluster = LocalCluster(
            n_workers=n_cores,
            threads_per_worker=1,
            processes=True,
        )
        client = Client(cluster)

        # ------------------------------------------------
        # --- Main body of the population tuning
        # ------------------------------------------------
        # --- Parameters specific to initialPopulationLearning
        if SRConstraints.constraintModules:  # --- if there are constraints
            extra_args = (
                "--learning_rate",
                f"{SRConfig.learning_rate_newborn}",
                "--iters",
                f"{int(SRConfig.totalItersInit / 2)}",
                "--reg2rmse",
                "0.0",
                "--constr2rmse",
                f"{SRConfig.constr2rmseMainRun}",
                "--sng2rmse",
                "0.0",
            )
        else:  # --- if there are no constraints
            extra_args = (
                "--learning_rate",
                f"{SRConfig.learning_rate_newborn}",
                "--iters",
                f"{int(SRConfig.totalItersInit / 2)}",
                "--reg2rmse",
                f"{SRConfig.reg2rmseMainRun}",
                "--constr2rmse",
                "0.0",
                "--sng2rmse",
                "0.0",
            )

        futures = self.submit_initial_tasks(client, initial, specific_params=extra_args)
        future_to_key = {fut: ind_id for ind_id, fut in futures.items()}
        ac = as_completed(futures.values())
        population = []
        for future in ac:
            key = future_to_key.pop(future, None)
            if key is not None:
                futures.pop(key, None)
            try:
                result = future.result()
                if result is None:
                    future.release()
                    continue
            except Exception as e:
                future.release()
                continue

            for text in result["logs"]:
                log(f"{text}", start_time=self.start_time, file_path=SRConfig.log_file)

            individual, readyNotTuned = self.process_individual(result)
            population.append(individual)

            future.release()

            # --- Pick individual from queue
            while queue:
                log(
                    f"Picked up next individual of ({queue[0].inPopulation}) from queue with id {queue[0].id}.",
                    start_time=self.start_time,
                    file_path=SRConfig.log_file,
                    level="DEBUG",
                )
                self.submit_next_task(
                    client,
                    queue,
                    futures,
                    ac,
                    future_to_key,
                    specific_params=extra_args,
                )
                break
        self.mainPopulation = population

        # --- Report populations sorted by complexity
        self.mainPopulation.sort(
            key=lambda ind: ind.last_perf.performance["complexity"]
        )
        log_population_state(
            population=self.mainPopulation,
            start_time=self.start_time,
            title="Initialization (after phase-II): Individuals in main population",
            file_path=SRConfig.outputNamePrefix + "initial_population_phase_II.txt",
        )

        for ind in self.mainPopulation:
            ind.saveIndividualPerformanceToFile(
                folderName=f"mainPopulationIndividuals",
                fileName=f"ind_{ind.id}",
                tickValue=f"{int(0)}",
                prefix="backprops_",
            )
        self._write_population_overview(
            self.mainPopulation,
            folderName="mainPopulationIndividuals",
            tickValue=f"{int(0)}",
            prefix="backprops_",
        )

        self._reset_main_population_processed()

        # --- 1. Cancel all futures if hard finish is triggered
        try:
            client.cancel(
                futures, force=True
            )  # --- force=True is important for stubborn tasks
            wait(
                futures, timeout="30s"
            )  # --- ensure scheduler processes the cancellation
            log(
                f"All futures canceled.",
                start_time=self.start_time,
                file_path=SRConfig.log_file,
                level="DEBUG",
            )
        except Exception:
            pass
        try:
            client.close()
        finally:
            cluster.close(timeout="30s")  # --- give workers time to die

    def computeQuantiles(self, population: list[Individual]):
        rmse_valids = [ind.last_perf.performance["valid_loss"] for ind in population]
        rmse_constrs = [ind.last_perf.performance["rmse_constr"] for ind in population]
        q_valid = SRConfig.multCoeff * np.quantile(rmse_valids, SRConfig.quantile)
        q_constr = SRConfig.multCoeff * np.quantile(rmse_constrs, SRConfig.quantile)

        return q_valid, q_constr

    def computeQuantilesFromMaximumPerformanceValues(
        self, population: list[Individual]
    ):
        if self.worst_valid is None:
            rmse_valids = [
                ind.last_perf.performance["valid_loss"] for ind in population
            ]
            rmse_constrs = [
                ind.last_perf.performance["rmse_constr"] for ind in population
            ]
            self.worst_valid = max(rmse_valids)
            self.worst_constr = max(rmse_constrs)
            log(
                f"Quantiles initialized using maximum performance values: worst_valid={self.worst_valid}, worst_constr={self.worst_constr}\n",
                start_time=self.start_time,
                file_path=SRConfig.log_file,
                level="HIGHLIGHT_CYAN",
            )
        else:
            rmse_valids = [
                ind.last_perf.performance["valid_loss"] for ind in population
            ]
            rmse_constrs = [
                ind.last_perf.performance["rmse_constr"] for ind in population
            ]
            worst_valid = max(rmse_valids)
            worst_constr = max(rmse_constrs)
            if worst_valid < self.worst_valid:
                log(
                    f"worst_valid updated: {self.worst_valid} -> {worst_valid}",
                    start_time=self.start_time,
                    file_path=SRConfig.log_file,
                    level="HIGHLIGHT_CYAN",
                )
                self.worst_valid = worst_valid
            if worst_constr < self.worst_constr:
                log(
                    f"worst_constr updated: {self.worst_constr} -> {worst_constr}",
                    start_time=self.start_time,
                    file_path=SRConfig.log_file,
                    level="HIGHLIGHT_CYAN",
                )
                self.worst_constr = worst_constr

        q_valid = SRConfig.multCoeff * self.worst_valid
        q_constr = SRConfig.multCoeff * self.worst_constr

        return q_valid, q_constr

    def find_replacement_candidate(self, offspring_ind: Individual) -> Individual:
        """
        Finds a replacement candidate in the main population for the given offspring individual.
        Args:
            offspring_ind (Individual): The offspring individual to evaluate.
        Returns:
            individual to replace
        """
        # --- find
        # ---   1) the maximum and minimum number of active nodes
        # ---   2) the best last_perf.performance["valid_loss"] and last_perf.performance["rmse_constr"]
        # --- in the main population
        max_nb_active_nodes = 0
        min_nb_active_nodes = 1e10
        best_valid_loss = 1e10
        best_rmse_constr = 1e10
        inds_with_max_nodes = []

        for ind in self.mainPopulation:
            if ind.last_perf.performance["nbOfActiveNodes"] > max_nb_active_nodes:
                max_nb_active_nodes = ind.last_perf.performance["nbOfActiveNodes"]
                inds_with_max_nodes = [ind]
            elif ind.last_perf.performance["nbOfActiveNodes"] == max_nb_active_nodes:
                inds_with_max_nodes.append(ind)
            if ind.last_perf.performance["nbOfActiveNodes"] < min_nb_active_nodes:
                min_nb_active_nodes = ind.last_perf.performance["nbOfActiveNodes"]
            # ---
            if ind.last_perf.performance["valid_loss"] < best_valid_loss:
                best_valid_loss = ind.last_perf.performance["valid_loss"]
            if ind.last_perf.performance["rmse_constr"] < best_rmse_constr:
                best_rmse_constr = ind.last_perf.performance["rmse_constr"]
            # --- Filter out individuals with either best_valid_loss or best_rmse_constr
            # --- TODO: Often causes stagnation. Think how to go around and do not loose solutions with best performance values.
            # inds_with_max_nodes = [
            #     ind
            #     for ind in inds_with_max_nodes
            #     if ind.last_perf.performance["valid_loss"] != best_valid_loss and ind.last_perf.performance["rmse_constr"] != best_rmse_constr
            # ]

        # --- decide if the offspring_ind can replace some of the individuals in inds_with_max_nodes
        if inds_with_max_nodes:
            if (
                max_nb_active_nodes - min_nb_active_nodes
            ) >= SRConfig.minNbDiffForQuantileReplacement:
                # --- large gap between the maximum and minimum subtopology size in the main population
                # if offspring_ind.last_perf.performance["nbOfActiveNodes"] < (min_nb_active_nodes + SRConfig.minNbDiffForQuantileReplacement):
                if (
                    offspring_ind.last_perf.performance["nbOfActiveNodes"]
                    < max_nb_active_nodes
                ):
                    # --- offspring is not too large
                    if len(inds_with_max_nodes) > 1:
                        _, dominated_set = splitPopulationByDomination(
                            inds_with_max_nodes,
                            front=0,
                            criteria=SubTopology.keysPerformance,
                        )
                        if dominated_set:
                            # --- among the individuals with max nodes, there is a dominated set
                            return SRConfig.r.choice(dominated_set)
                        else:
                            # --- all individuals with max nodes are mutually non-dominated
                            return inds_with_max_nodes[0]
                    else:
                        # --- there is only one individual with max nodes
                        return inds_with_max_nodes[0]
            elif (
                offspring_ind.last_perf.performance["nbOfActiveNodes"]
                <= max_nb_active_nodes
            ):
                # --- the gap between the maximum and minimum subtopology size is small
                return SRConfig.r.choice(inds_with_max_nodes)

        return None

    def addOffspringToPopulationNever(self, offspring_ind: Individual) -> bool:
        return False

    def addOffspringToPopulation(self, offspring_ind: Individual) -> bool:
        """
        Try to add an offspring individual to the main population.

        Rules (in order):

        1) Offspring must be an adult; otherwise do nothing.

        2) Check if the offspring could be added to the mainPopulation when it is not full.

        3a) Look for adults in the main population that:
             - are at least as old as the offspring, and used the same dataset,
             - and are dominated by the offspring (using SubTopology.keysAll on last performance).
           If any are found, replace the most similar/the biggest dominated individual with the offspring.

        3b) If none are dominated:
             - If the main population is full:
                 I.  a) Check that offspring's valid_loss and rmse_constr are not worse than the quantiles.
                     b) Check that the offspring's nbOfActiveNodes is not *larger* than the maximum
                        in the main population.
                     c) If both checks pass, replace one of the individuals with the maximum number
                        of active nodes. If multiple candidates tie:
                           - Split by domination (SubTopology.keysPerformance), and pick randomly
                             among the dominated set.
                II.  a) Check that offspring's either valid_loss or rmse_constr is not worse than the quantiles.
                     b) Check that the number of nodes is smaller than the minimum in the main population.
                     c) If both checks pass, replace one of the individuals with the maximum number
                        of active nodes. If multiple candidates tie:
                           - Split by domination (SubTopology.keysPerformance), and pick randomly
                             among the dominated set.
             - Otherwise, do nothing.

        Returns:
            True if the offspring was added to the main population, False otherwise.
        """
        # ----------------------------------------------------------------------------
        # --- 1) do not add to the main population until offspring becomes an adult
        # ----------------------------------------------------------------------------
        if not offspring_ind.adult:
            return False

        # -------------------------------------------------------------------------------------
        # --- 2) check if offspring could be added to the mainPopulation when it is not full
        # -------------------------------------------------------------------------------------
        if len(
            self.mainPopulation
        ) < SRConfig.main_pop_size and self._offspring_is_eligible_to_notfull_mainpop(
            offspring_ind
        ):
            offspring_ind.inPopulation = "mainPopulation"
            self.mainPopulation.append(offspring_ind)
            self.updateArchive([offspring_ind])
            self.offspring.remove(offspring_ind)

            # # -----------------------------------------------
            # # --- mainPopulation changed ==> assign fronts
            # # -----------------------------------------------
            # self.mainPopulation = nonDominatedSorting(
            #     population=copy.deepcopy(self.mainPopulation),
            #     finalSize=len(self.mainPopulation),
            #     criteria=SubTopology.keysAll,
            # )

            # --- Report populations sorted by complexity and id
            self.mainPopulation.sort(
                key=lambda ind: ind.last_perf.performance["complexity"]
            )
            self.offspring.sort(key=lambda ind: ind.id)
            log(
                f"Offspring individual goes to unfull mainPopulation: offspring_ind {offspring_ind.id} with {offspring_ind.backprop_iters} backprops_iters.",
                self.start_time,
                file_path=SRConfig.log_file,
                level="HIGHLIGHT_CYAN",
            )
            return True

        # ----------------------------------------------------------------------------
        # --- 3) check for dominated individuals in the mainPopulation
        # ----------------------------------------------------------------------------
        dominated = []  # list of individuals that are dominated by offspring_ind
        for ind in self.mainPopulation:
            if not ind.adult:
                continue  # --- do not compare until the individual is not adult
            if getattr(ind, SRConfig.ageToConsider) < getattr(
                offspring_ind, SRConfig.ageToConsider
            ):
                continue  # --- do not compare if the individual is younger than offspring_ind
            # Dataset-aware dominance: the explicit math.isclose timestamp
            # guard that used to live here has been folded into
            # _dataset_aware_dominance. Cross-dataset pairs are now resolved
            # by the "current always wins" rule rather than being skipped.
            firstDominates, _, firstEqualsSecond = self._dataset_aware_dominance(
                offspring_ind.last_perf,
                ind.last_perf,
                criteria=SubTopology.keysAll,
            )
            if firstDominates and not firstEqualsSecond:
                dominated.append(ind)

        ind_to_replace = None
        reason = None
        if dominated:
            if SRConfig.replaceDominatedBy == "similarity":
                similarities = individualPerformanceSimilarity(offspring_ind, dominated)
                most_similar_idx = similarities.index(min(similarities))
                ind_to_replace = dominated[most_similar_idx]
            elif SRConfig.replaceDominatedBy == "size":
                nb_nodes = [
                    ind.last_perf.performance["nbOfActiveNodes"] for ind in dominated
                ]
                biggest_idx = nb_nodes.index(max(nb_nodes))
                ind_to_replace = dominated[biggest_idx]
            reason = "dominance"

        if not ind_to_replace:
            log(
                f"Offspring individual does not dominate any individual in the main pop, and does not satisfy the quantile condition: offspring_ind {offspring_ind.id} with {offspring_ind.backprop_iters} backprops_iters.\n"
                f"Offspring is not added to the main population, but stays in offspring.",
                start_time=self.start_time,
                file_path=SRConfig.log_file,
                level="DEBUG",
            )
            return False

        # ---- handle individual to be replaced from mainPopulation
        ind_to_replace.inPopulation = None
        self.mainPopulation.remove(ind_to_replace)
        if ind_to_replace.id in self.mainPopulationProcessed:
            del self.mainPopulationProcessed[ind_to_replace.id]

        # ---- handle offspring individual added to mainPopulation
        offspring_ind.inPopulation = "mainPopulation"
        self.mainPopulation.append(offspring_ind)
        self.updateArchive([offspring_ind])
        self.offspring.remove(offspring_ind)

        # # -----------------------------------------------
        # # --- mainPopulation changed ==> assign fronts
        # # -----------------------------------------------
        # self.mainPopulation = nonDominatedSorting(
        #     population=copy.deepcopy(self.mainPopulation),
        #     finalSize=len(self.mainPopulation),
        #     criteria=SubTopology.keysAll,
        # )

        # --- Report populations sorted by complexity and id
        self.mainPopulation.sort(
            key=lambda ind: ind.last_perf.performance["complexity"]
        )
        self.offspring.sort(key=lambda ind: ind.id)
        log(
            f"Offspring individual replaces mainPopulation individual: offspring_ind {offspring_ind.id} with {offspring_ind.backprop_iters} backprops_iters, replacement {ind_to_replace.id} ({reason}).\n",
            self.start_time,
            file_path=SRConfig.log_file,
            level="HIGHLIGHT_CYAN",
        )
        return True

    def addNewIndToOffspring(self, individual: Individual) -> bool:
        """
        The individual is added to the offspring population only if the offspring population size is less than SRConfig.offspring_pop_size.
        """
        if len(self.offspring) < SRConfig.offspring_pop_size:
            individual.inPopulation = "offspring"
            self.offspring.append(individual)
            self.offspring.sort(key=lambda ind: ind.id)
            self.updateArchive([individual])
            log(
                f"Added individual {individual.id} to offspring.",
                self.start_time,
                level="DEBUG",
            )
            # log_population_state(
            #     population=self.offspring,
            #     start_time=self.start_time,
            #     file_path=SRConfig.log_file,
            #     title="Individuals in offspring population",
            # )
            return True
        else:
            log(
                f"Offspring population is full. Individual {individual.id} not added.",
                self.start_time,
                file_path=SRConfig.log_file,
                level="HIGHLIGHT_CYAN",
            )
            return False

    def _dominates_any_in_offspring(self, individual: Individual) -> bool:
        age = getattr(individual, SRConfig.ageToConsider)
        for other in self.offspring:
            if not other.adult:
                # --- do not compare until the individual is not an adult
                continue
            if not age in other.perf_history.keys():
                # --- do not compare if the other individual is younger or does not have tracked performance at the same age
                continue
            # --- check dominance
            # Dataset-aware: cross-dataset pairs are resolved by the
            # "current always wins" rule inside _dataset_aware_dominance,
            # so the previous math.isclose timestamp guard is unnecessary.
            # We compare last_perf vs last_perf (consistent), which is the
            # same pair the dominance check below uses for its metrics -
            # the original code mixed perf_history[age].dataTimestamp on
            # the skip side with other.last_perf.performance on the
            # comparison side; this fixes that subtle inconsistency.
            firstDominates, _, firstEqualsSecond = self._dataset_aware_dominance(
                individual.last_perf,
                other.last_perf,
                criteria=SubTopology.keysPerfCplx,
            )
            if firstDominates and not firstEqualsSecond:
                return other
        return None

    def addNewIndToOffspringIfDominates(self, individual: Individual) -> bool:
        """
        The individual is added to the offspring population only if
          - the offspring population size is less than SRConfig.offspring_pop_size
          - or if the individual dominates any individual in the offspring population.
        """
        if len(self.offspring) < SRConfig.offspring_pop_size:
            individual.inPopulation = "offspring"
            self.offspring.append(individual)
            self.offspring.sort(key=lambda ind: ind.id)
            self.updateArchive([individual])
            log(
                f"Added individual {individual.id} to offspring which is not full yet.",
                self.start_time,
                file_path=SRConfig.log_file,
                level="HIGHLIGHT_CYAN",
            )
            # log_population_state(
            #     population=self.offspring,
            #     start_time=self.start_time,
            #     file_path=SRConfig.log_file,
            #     title=f"Individuals in offspring population after adding, to not full population, individual {individual.id}",
            # )
            return True

        dominated = self._dominates_any_in_offspring(individual)
        if dominated:
            self.offspring.remove(dominated)
            dominated.inPopulation = None
            log(
                f"Individual {dominated.id} in offspring population is dominated by new individual {individual.id}; removed from offspring population.",
                self.start_time,
                file_path=SRConfig.log_file,
                level="HIGHLIGHT_CYAN",
            )
            individual.inPopulation = "offspring"
            self.offspring.append(individual)
            log(
                f"Added individual {individual.id} to offspring population replacing dominated {dominated.id}.",
                self.start_time,
                file_path=SRConfig.log_file,
                level="HIGHLIGHT_CYAN",
            )
            self.offspring.sort(key=lambda ind: ind.id)
            self.updateArchive([individual])
            # log_population_state(
            #     population=self.offspring,
            #     start_time=self.start_time,
            #     file_path=SRConfig.log_file,
            #     title=f"Individuals in offspring population after removing dominated {dominated.id}",
            # )
            return True
        else:
            log(
                f"Offspring population is full. Individual {individual.id} not added.",
                self.start_time,
                file_path=SRConfig.log_file,
                level="HIGHLIGHT_CYAN",
            )
            return False

    def _check_dominance_and_remove(self, ind: Individual, other: Individual) -> bool:
        if not ind in self.mainPopulation:
            return False
        if not other.adult:
            return False  # do not compare until the individual is not an adult

        if not ind.subjectToTune or getattr(  # --- individual is not being tuned -> compare on last performance also with older individuals
            other, SRConfig.ageToConsider
        ) < getattr(
            ind, SRConfig.ageToConsider
        ):  # --- other individual is younger
            other_perf_to_consider = other.last_perf
        else:
            age = getattr(ind, SRConfig.ageToConsider)
            if not age in other.perf_history.keys():
                return False
            other_perf_to_consider = other.perf_history[age]

        # Dataset-aware dominance: the explicit math.isclose timestamp guard
        # that used to short-circuit cross-dataset pairs has been folded into
        # _dataset_aware_dominance. Under the new rule:
        #   - both perfs current  -> normal getMutualDominance result
        #   - ind current, other stale -> ind wins (secondDominates=False, no removal)
        #   - ind stale, other current -> other wins (secondDominates=True, ind removed)
        #   - both stale          -> incomparable (no removal)
        _, secondDominates, firstEqualsSecond = self._dataset_aware_dominance(
            ind.last_perf,
            other_perf_to_consider,
            criteria=SubTopology.keysAll,
        )
        if secondDominates and not firstEqualsSecond:
            self.mainPopulation.remove(ind)
            ind.inPopulation = None
            ind_age = getattr(ind, SRConfig.ageToConsider)
            other_age = getattr(other, SRConfig.ageToConsider)
            if ind.subjectToTune and other_age > ind_age:
                other_age = ind_age

            # # -----------------------------------------------
            # # --- mainPopulation changed ==> assign fronts
            # # -----------------------------------------------
            # self.mainPopulation = nonDominatedSorting(
            #     population=copy.deepcopy(self.mainPopulation),
            #     finalSize=len(self.mainPopulation),
            #     criteria=SubTopology.keysAll,
            # )

            # --- Report populations sorted by complexity
            self.mainPopulation.sort(
                key=lambda ind: ind.last_perf.performance["complexity"]
            )
            log(
                f"Individual {ind.id} is dominated by {other.id} in main population.\n"
                f"Performance of individual {ind.id} at {SRConfig.ageToConsider} {ind_age}: {ind.last_perf.performance},\n"
                f"Performance of individual {other.id} at {SRConfig.ageToConsider} {other_age}: {other.perf_history[other_age].performance}.\n"
                f"Removed individual {ind.id} from main population.",
                self.start_time,
                file_path=SRConfig.log_file,
            )
            log_population_state(
                population=self.mainPopulation,
                start_time=self.start_time,
                title="Individuals in main population",
                file_path=SRConfig.log_file,
            )
            return True
        return False

    def keepIndInMainPopulationConditional(self, ind: Individual) -> bool:
        """
        The individual is kept in the main population if it is not adult or if the main population is not full.
        Otherwise, we check if there is any other individual from the main population that dominates the individual and was trained on the same data.
        If the other individual is younger, we consider their last saved performance. If it is older, we compare them at their common age level.
        """
        if not ind.adult:
            return True
        if len(self.mainPopulation) < SRConfig.main_pop_size:
            return True
        for other in self.mainPopulation:
            if self._check_dominance_and_remove(ind, other):
                return False
        return True

    def keepIndInMainPopulationAlways(self, ind: Individual) -> bool:
        """
        The individual is always kept in the main population.
        """
        return True

    def keepIndInOffspringAlways(self, ind: Individual) -> bool:
        """
        The individual is always kept in the offspring population.
        """
        return True

    def keepIndInOffspringConditional(self, ind: Individual) -> bool:
        """
        The individual is kept in the offspring population if it is not adult.
        The individual is removed from the offspring population if it's subjectToTune is False.
        We check if there is any other individual from the offspring population of the same age that dominates the individual and was trained on the same data.
        """

        if not ind.adult:
            return True

        if not ind.subjectToTune:
            self.offspring.remove(ind)
            ind.inPopulation = None
            log(
                f"Individual {ind.id} is not subject to tune and will be removed from offspring population.",
                self.start_time,
                file_path=SRConfig.log_file,
                level="DEBUG",
            )
            self.offspring.sort(key=lambda ind: ind.id)
            # log_population_state(
            #     population=self.offspring,
            #     start_time=self.start_time,
            #     file_path=SRConfig.log_file,
            #     title=f"Individuals in offspring population after removing not subject to tune {ind.id}",
            # )
            return False

        # -- check dominance
        age = getattr(ind, SRConfig.ageToConsider)
        for other in self.offspring:
            if not other.adult:
                continue  # do not compare until the individual is not an adult
            if not age in other.perf_history.keys():
                continue  # do not compare if the other individual is younger or does not have tracked performance at the same age
            # Dataset-aware dominance: the explicit math.isclose timestamp
            # guard that used to skip cross-dataset pairs is now folded into
            # _dataset_aware_dominance. Under the new rule a stale ind
            # is dominated by every current other.perf_history[age] entry
            # (and triggers removal); a current ind cannot be dominated by
            # any stale entry (no removal).
            _, secondDominates, firstEqualsSecond = self._dataset_aware_dominance(
                ind.last_perf,
                other.perf_history[age],
                criteria=SubTopology.keysAll,
            )
            if secondDominates and not firstEqualsSecond:
                self.offspring.remove(ind)
                ind.inPopulation = None
                log(
                    f"Individual {ind.id} is dominated by {other.id} in offspring population at {SRConfig.ageToConsider} {age}.\n"
                    f"Performance of individual {ind.id} at {SRConfig.ageToConsider} {age}: {ind.last_perf.performance},\n"
                    f"Performance of individual {other.id} at {SRConfig.ageToConsider} {age}: {other.perf_history[age].performance}.\n"
                    f"Removed individual {ind.id} with {ind.backprop_iters} backprop_iters from offspring population due to dominance.",
                    self.start_time,
                    level="DEBUG",
                )
                self.offspring.sort(key=lambda ind: ind.id)
                # log_population_state(
                #     population=self.offspring,
                #     start_time=self.start_time,
                #     file_path=SRConfig.log_file,
                #     title=f"Individuals in offspring population after removing dominated {ind.id}",
                # )
                return False

        return True

    def submit_initial_tasks(self, client, initial, specific_params: tuple = ()):
        """Submit initial tasks to the Dask client."""
        futures = {}
        # --- Add specific parameters for train_nn()
        train_nn_params = self.cli_args + specific_params
        for ind in initial:
            log(
                f"Submitted individual {ind.id} for tuning in initial task.",
                self.start_time,
                file_path=SRConfig.log_file,
                level="DEBUG",
            )
            self.currentlyTuned.add(ind.id)
            futures[ind.id] = client.submit(
                run_parallel_finetune,
                train_nn_params,
                ind.last_perf.activeNodesCoordinates,
                ind.last_perf.nn_weights,
                ind.last_perf.performance,
                ind.id,
                ind.inPopulation,
                start_time=self.start_time,
                backpropIters=ind.backprop_iters,
                lastPruningBackprops=ind.lastPruningBackprops,
                pure=False,
            )
            self.numOfBackpropTunings += 1
            try:
                self.totalBackpropsUsed += int(
                    specific_params[specific_params.index("--iters") + 1]
                )
                log(
                    f"Total backprops used ({self.numOfBackpropTunings}): {self.totalBackpropsUsed}.",
                    self.start_time,
                    file_path=SRConfig.log_file,
                    level="HIGHLIGHT_GREEN",
                )
            except (ValueError, IndexError):
                log(
                    f"Error parsing '--iters' parameter.",
                    self.start_time,
                    file_path=SRConfig.log_file,
                    level="ERROR",
                )
        return futures

    def submit_next_task(
        self, client, queue, futures, ac, futures_to_id, specific_params: tuple = ()
    ):
        """Submit the next task from the queue."""
        # --- Add specific parameters for train_nn()
        train_nn_params = self.cli_args + specific_params
        next_ind = queue.popleft()
        self.currentlyTuned.add(next_ind.id)
        new_future = client.submit(
            run_parallel_finetune,
            train_nn_params,
            next_ind.last_perf.activeNodesCoordinates,
            next_ind.last_perf.nn_weights,
            next_ind.last_perf.performance,
            next_ind.id,
            next_ind.inPopulation,
            start_time=self.start_time,
            backpropIters=next_ind.backprop_iters,
            lastPruningBackprops=next_ind.lastPruningBackprops,
            pure=False,
        )

        # --- Remove stale entries for this individual before inserting new future
        old_future = futures.pop(next_ind.id, None)
        if old_future is not None:
            futures_to_id.pop(old_future, None)

        # --- Now insert the new future
        futures[next_ind.id] = new_future
        futures_to_id[new_future] = next_ind.id
        ac.add(new_future)

        # --- DIAGNOSTIC
        log(
            f"[DIAG] After ac.add: ac.count()={ac.count()}, futures={list(futures.keys())}, currentlyTuned={self.currentlyTuned}, new_future.status={new_future.status}",
            self.start_time,
            file_path=SRConfig.log_file,
            level="HIGHLIGHT_RED",
        )
        self.numOfBackpropTunings += 1

        try:
            self.totalBackpropsUsed += int(
                specific_params[specific_params.index("--iters") + 1]
            )
            log(
                f"Next task submitted, total backprops used ({self.numOfBackpropTunings}): {self.totalBackpropsUsed}.",
                self.start_time,
                file_path=SRConfig.log_file,
                level="HIGHLIGHT_GREEN",
            )
        except (ValueError, IndexError):
            log(
                f"Error parsing '--iters' parameter.",
                self.start_time,
                file_path=SRConfig.log_file,
                level="ERROR",
            )

    def process_individual(self, result):
        """
        Update the performance and weights of an individual.
        Check if individual stagnates or becomes an adult.
        """
        # --- Find all the individuals that are only ageing and are waiting for this individual to finish tuning
        readyNotTuned: list[Individual] = []
        if result["id"] in self.currentlyTuned:
            self.currentlyTuned.remove(result["id"])

        for ind in list(self.waitingNotTuned.keys()):
            if result["id"] in self.waitingNotTuned[ind]:
                self.waitingNotTuned[ind].remove(result["id"])
            if not self.waitingNotTuned[ind]:
                readyNotTuned.append(ind)
                del self.waitingNotTuned[ind]

        # --- Find the individual in the main population or offspring and update its performance
        individual = next(
            (
                ind
                for ind in self.mainPopulation + self.offspring
                if ind.id == result["id"]
            ),
            None,
        )
        if individual:
            # --- Update individual performance history
            individual.age += result["iters"]
            individual.backprop_iters += result["iters"]
            # --- Carry back the pruning watermark advanced inside train_nn so the
            #     individual's next round measures the pruning interval correctly.
            individual.lastPruningBackprops = result.get(
                "lastPruningBackprops", individual.lastPruningBackprops
            )
            history_update = result["historyUpdate"]
            individual.last_perf = history_update
            individual.perf_history[getattr(individual, SRConfig.ageToConsider)] = (
                history_update
            )
            individual.last_output = result["output"].detach().cpu().numpy()
            individual.data_timestamp = result["historyUpdate"].dataTimestamp
            individual.expr = result["expr"]
            individual.simpleExpr = result["simpleExpr"]
            log(
                f"Updated individual {individual.id} with new performance history, "
                f"{SRConfig.ageToConsider} of individual: {getattr(individual, SRConfig.ageToConsider)}.",
                self.start_time,
                file_path=SRConfig.log_file,
                level="HIGHLIGHT_GREEN",
            )

            # --- Save 'mainpopulation' and 'offspring' individuals to file after each backprop update
            if (
                SRConfig.genealogyPath.strip()
                and (
                    individual.inPopulation == "mainPopulation"
                    or individual.inPopulation == "offspring"
                )
                # and individual.last_perf.performance["nbOfImprovements"] != 0
            ):
                # --- something changed in the performance -> save the individual
                individual.saveIndividualToFile(
                    filePath=f"{SRConfig.genealogyPath}ind={individual.id}_backprops={individual.backprop_iters}.txt"
                )

            # --- Check if individual stagnates
            if individual.inPopulation == "mainPopulation" and not SRConfig.standard_EA:
                max_noimprovment_epochs = SRConfig.max_noimprovment_epochs_main
            elif individual.inPopulation == "mainPopulation" and SRConfig.standard_EA:
                if SRConfig.ea_ignores_stagnation:
                    max_noimprovment_epochs = (
                        sys.maxsize
                    )  # --- effectively disable stagnation check for main population in standard EA
                else:
                    max_noimprovment_epochs = SRConfig.max_noimprovment_epochs_main
            else:
                max_noimprovment_epochs = SRConfig.max_noimprovment_epochs_offspring
            k = sum(
                [
                    individual.perf_history[ph_id].performance["nbOfImprovements"]
                    for ph_id in list(individual.perf_history.keys())[
                        -max_noimprovment_epochs:
                    ]
                ]
            )
            if len(individual.perf_history) >= max_noimprovment_epochs and k == 0:
                individual.subjectToTune = False
                log(
                    f"SubjectToTune of individual {individual.inPopulation}/{individual.id} is set to False with backprop_iters={individual.backprop_iters}.",
                    self.start_time,
                    file_path=SRConfig.log_file,
                    level="DEBUG",
                )
                if individual.inPopulation == "mainPopulation":
                    self.adjust_subjecttotune_in_mainpop()
            else:
                individual.age_last_change_backprop = individual.age

            # --- Check if individual becomes an adult
            if (
                not individual.adult
                and getattr(individual, SRConfig.ageToConsider) >= SRConfig.th_adult
            ):
                individual.adult = True
                log(
                    f"Individual {individual.id} has become an adult at {SRConfig.ageToConsider}: {getattr(individual, SRConfig.ageToConsider)}.",
                    self.start_time,
                    file_path=SRConfig.log_file,
                    level="HIGHLIGHT_CYAN",
                )

            # --- Update archive
            if (
                individual.inPopulation == "offspring"
                and individual.last_perf.performance["nbOfImprovements"] != 0
            ):
                self.updateArchive([individual])
                self.archive.sort(
                    key=lambda ind: ind.last_perf.performance["complexity"]
                )

            if individual.id in self.mainPopulationProcessed:
                self.mainPopulationProcessed[individual.id] += 1

        else:
            self.totalBackpropsUsed -= result["iters"]
            log(
                f"Individual with id {result['id']} not found in either population; totalBackpropsUsed decremented.",
                self.start_time,
                file_path=SRConfig.log_file,
                level="DEBUG",
            )
        return individual, readyNotTuned

    def _generate_new_add_to_queue(self, queue, triggerId: int = -1):
        """
        Generate a new individual by crossover or mutation and add it to the queue for further tuning.
        """
        new_individuals = self.crossover_mutation_reproduction(
            regularize=False, triggerId=triggerId
        )
        if not new_individuals:
            return
        for new_ind in new_individuals:
            if new_ind is not None:
                addToQueue, method = self._add_to_queue(new_ind)
                if addToQueue:
                    if method == "appendLeft":
                        queue.appendleft(new_ind)
                    else:
                        queue.append(new_ind)
                    log(
                        f"Individual {new_ind.id} added to queue for further tuning.",
                        self.start_time,
                        file_path=SRConfig.log_file,
                        level="HIGHLIGHT_CYAN",
                    )
                else:
                    log(
                        f"Individual {new_ind.id} created by crossover is not used.",
                        self.start_time,
                        file_path=SRConfig.log_file,
                        level="DEBUG",
                    )

    def _add_to_queue(self, individual: Individual):
        """
        Returns 'addToQueue addWhere':
           addToQueue: True / False
           addWhere: 'appendLeft' / 'append'
        """
        # --- do not add to queue if individual has too few active nodes and remove it from the population
        if (
            individual.last_perf.performance["nbOfActiveNodes"]
            < SRConfig.minNbOfActiveNodes
        ):
            log(
                f"Individual {individual.id} has too few active nodes. It is not added to the queue for further tuning.",
                self.start_time,
                file_path=SRConfig.log_file,
                level="HIGHLIGHT_CYAN",
            )
            if individual.inPopulation:
                log(
                    f"Individual {individual.id} is being removed from {individual.inPopulation}.",
                    self.start_time,
                    file_path=SRConfig.log_file,
                    level="HIGHLIGHT_CYAN",
                )
                if individual.inPopulation == "offspring":
                    self.offspring.remove(individual)
                    individual.inPopulation = None
                elif individual.inPopulation == "mainPopulation":
                    self.mainPopulation.remove(individual)
                    individual.inPopulation = None
                    if individual.id in self.mainPopulationProcessed:
                        del self.mainPopulationProcessed[individual.id]
                    # # -----------------------------------------------
                    # # --- mainPopulation changed ==> assign fronts
                    # # -----------------------------------------------
                    # self.mainPopulation = nonDominatedSorting(
                    #     population=copy.deepcopy(self.mainPopulation),
                    #     finalSize=len(self.mainPopulation),
                    #     criteria=SubTopology.keysAll,
                    # )

            return False, None

        addedToOffspring = False

        # --- check if a NEWLY CREATED individual can be added to offspring population
        if not individual.inPopulation in ["mainPopulation", "offspring"]:
            addedToOffspring = getattr(self, SRConfig.addNewIndToOffspring)(individual)
            if addedToOffspring:
                return True, "appendLeft"
            else:
                return False, None

        # --- check if individual should be kept in main population
        if individual.inPopulation == "mainPopulation":
            if not getattr(self, SRConfig.keepIndInMainPopulation)(individual):
                if individual.id in self.mainPopulationProcessed:
                    del self.mainPopulationProcessed[individual.id]
                return False, None
            else:
                return True, "append"

        # --- check if individual should be kept in offspring population
        keptInOffspring = False
        if individual.inPopulation == "offspring" and not addedToOffspring:
            keptInOffspring = getattr(self, SRConfig.keepIndInOffspring)(individual)

        # --- check if offspring could be added to main population
        if addedToOffspring or keptInOffspring:
            getattr(self, SRConfig.addOffspringToPopulation)(individual)

        # --- check, if the number of backprop iterations of the offspring does not exceed the threshold
        if (
            individual.inPopulation == "offspring"
            and individual.backprop_iters >= SRConfig.max_backprop_iters_offspring
        ):
            self.offspring.remove(individual)
            individual.inPopulation = None
            keptInOffspring = False
            log(
                f"Individual {individual.id} has reached the maximum number of backprop iterations ({SRConfig.max_backprop_iters_offspring}); It is removed from the offspring population.",
                self.start_time,
                file_path=SRConfig.log_file,
                level="HIGHLIGHT_CYAN",
            )

        # --- check, if the offspring is still subject to tune
        if individual.inPopulation == "offspring" and not individual.subjectToTune:
            self.offspring.remove(individual)
            individual.inPopulation = None
            keptInOffspring = False
            log(
                f"Individual {individual.id} is not subject to tune and therefore is removed from the offspring population.",
                self.start_time,
                level="HIGHLIGHT_CYAN",
            )

        addToQueue = addedToOffspring or keptInOffspring
        return addToQueue, "append"

    def _add_eligible_offsprings_to_main_if_not_full(self):
        """ "
        If the main population is not full, create non-dominated front of offsprings based on valid loss and constraints.
        Add individuals from the front with the which
           1) have the number of active nodes less than or equal to the maximum in the main population,
           2) have 'valid_loss' and 'rmse_constr' less than or equal to the quantiles of the main population.
        If there are more of them, take them in random order.
        """

        main_max_size = max(
            [
                ind.last_perf.performance["nbOfActiveNodes"]
                for ind in self.mainPopulation
            ]
        )
        offspring_adults = [
            ind
            for ind in self.offspring
            if ind.adult
            and ind.last_perf.performance["nbOfActiveNodes"] <= main_max_size
        ]
        already_added_one = False

        while len(self.mainPopulation) < SRConfig.main_pop_size and offspring_adults:

            # q_valid, q_constr = self.computeQuantiles(population=self.mainPopulation)
            q_valid, q_constr = self.computeQuantilesFromMaximumPerformanceValues(
                population=self.mainPopulation
            )
            nonDominatedOffspringAdults, _ = splitPopulationByDomination(
                offspring_adults, front=0, criteria=SubTopology.keysPerformance
            )
            nonDominatedOffspringAdults.sort(
                key=lambda ind: ind.last_perf.performance["nbOfActiveNodes"]
            )
            min_offspring_nb_of_active_nodes = nonDominatedOffspringAdults[
                0
            ].last_perf.performance["nbOfActiveNodes"]
            candidates = [
                ind
                for ind in nonDominatedOffspringAdults
                if ind.last_perf.performance["nbOfActiveNodes"]
                == min_offspring_nb_of_active_nodes
            ]
            # candidates = [ind for ind in nonDominatedOffspringAdults if ind.last_perf.performance["nbOfActiveNodes"] <= main_max_size]

            while True:
                if not candidates:
                    new_ind = None
                    break

                new_ind = SRConfig.r.choice(candidates)
                candidates.remove(new_ind)
                if (
                    new_ind.last_perf.performance["valid_loss"] <= q_valid
                    and new_ind.last_perf.performance["rmse_constr"] <= q_constr
                ):
                    break  # --- Found one individual to add

            if not new_ind and not already_added_one:
                log(
                    f"No offspring individual can be added to the main population.\n"
                    f"Quantile of valid RMSE in main population: {q_valid}, quantile of constraint RMSE in main population: {q_constr}.\n",
                    self.start_time,
                    file_path=SRConfig.log_file,
                    level="DEBUG",
                )
                return

            if not new_ind:
                return

            new_ind.inPopulation = "mainPopulation"
            self.mainPopulation.append(new_ind)
            self.updateArchive([new_ind])
            self.offspring.remove(new_ind)
            offspring_adults.remove(new_ind)
            already_added_one = True

            # # -----------------------------------------------
            # # --- mainPopulation changed ==> assign fronts
            # # -----------------------------------------------
            # self.mainPopulation = nonDominatedSorting(
            #     population=copy.deepcopy(self.mainPopulation),
            #     finalSize=len(self.mainPopulation),
            #     criteria=SubTopology.keysAll,
            # )

            # --- Report populations sorted by complexity
            self.mainPopulation.sort(
                key=lambda ind: ind.last_perf.performance["complexity"]
            )
            self.offspring.sort(key=lambda ind: ind.id)
            log(
                f"Max capacity of main population is not filled.\n"
                f"Offspring individual {new_ind.id} with {new_ind.backprop_iters} backprop_iters is added to the main population and removed from the offsprings.\n",
                self.start_time,
                file_path=SRConfig.log_file,
                level="HIGHLIGHT_CYAN",
            )
            # log_population_state(
            #     population=self.mainPopulation,
            #     start_time=self.start_time,
            #     file_path=SRConfig.log_file,
            #     title="Individuals in main population",
            # )
            # log_population_state(
            #     population=self.offspring,
            #     start_time=self.start_time,
            #     file_path=SRConfig.log_file,
            #     title="Individuals in offspring population",
            # )
        return

    def _offspring_is_eligible_to_notfull_mainpop(self, ind: Individual) -> bool:
        """ "
        If the main population is not full, add the individual, if it
           1) has the number of active nodes less than or equal to the maximum in the main population,
           2) has 'valid_loss' and 'rmse_constr' less than or equal to the quantiles of the main population.
        If there are more of them, take them in random order.
        """
        if not ind.adult:
            return False
        # --- check individual complexity
        max_acceptable_size = np.median(
            [
                ind.last_perf.performance["nbOfActiveNodes"]
                for ind in self.mainPopulation
            ]
        )
        if ind.last_perf.performance["nbOfActiveNodes"] > max_acceptable_size:
            return False
        # --- check individual performance against q-values
        q_valid, q_constr = self.computeQuantilesFromMaximumPerformanceValues(
            population=self.mainPopulation
        )
        if (
            ind.last_perf.performance["valid_loss"] <= q_valid
            and ind.last_perf.performance["rmse_constr"] <= q_constr
        ):
            return True
        return False

    def _handle_not_subject_to_tune(self, ind, queue):
        # --- TODO another option is to return back to tuning once there was a dataset change
        # print(f'self.data_timestamp: {self.data_timestamp}')
        # print(f'os.path.getmtime(SRConfig.train_data): {os.path.getmtime(SRConfig.train_data)}')
        # print(f'ind[{ind.id}].last_perf.dataTimestamp: {ind.last_perf.dataTimestamp}')

        # --- Keep the manager's loaded data in sync with the file on disk.
        #     NOTE: exact != comparison. math.isclose(..., rel_tol=1e-9) was wrong here:
        #     at Unix-timestamp magnitudes (~1.78e9) the effective tolerance was ~1.78 s,
        #     so any file change happening within ~2 s of the previous reload was silently
        #     treated as "no change" and reload_data() never fired. Both sides of this
        #     comparison come straight from os.path.getmtime, so they are bit-identical
        #     when the file is unchanged - no tolerance is needed.
        if self.data_timestamp != os.path.getmtime(SRConfig.train_data):
            self.reload_data()

        # --- Recompute this individual's performance if its last_perf was computed
        #     against older data than what the manager currently holds. Gating on the
        #     per-individual timestamp (rather than on file mtime vs self.data_timestamp)
        #     ensures the recompute still fires when another path - e.g. the reload at
        #     crossover_mutation_reproduction (~line 584) - already brought the manager
        #     in sync with disk before this method ran.
        if ind.last_perf.dataTimestamp != self.data_timestamp:
            log(
                f"Data has changed since last performance of individual {ind.id}. Recomputing performance.",
                start_time=self.start_time,
                file_path=SRConfig.log_file,
            )
            st = SubTopology.createSubtopologyFromParameters(
                activeNodesCoordinates=ind.last_perf.activeNodesCoordinates,
                nnWeights=ind.last_perf.nn_weights,
                losses=ind.last_perf.performance,
            )
            try:
                _, nbOfImprovements = st.train_nn(
                    learningSteps=0,
                    learningRate=SRConfig.learning_rate_newborn,
                    regularize=True,
                    clipWeights=True,
                    backpropItersBefore=ind.backprop_iters,
                    lastPruningBackprops=ind.lastPruningBackprops,
                )
                # learningSteps=0 cannot cross the pruning interval, but carry
                # back the watermark for consistency.
                ind.lastPruningBackprops = st.lastPruningBackprops
            except Exception as e:
                log(
                    f"Exception in child.train_nn: {e}",
                    start_time=self.start_time,
                    file_path=SRConfig.log_file,
                )
            ind.last_perf.performance["valid_loss"] = st.losses["valid_loss"]
            ind.last_perf.performance["rmse_constr"] = st.losses["rmse_constr"]
            ind.last_perf.dataTimestamp = self.data_timestamp
            ind.last_output = (
                st.get_subtopology_output(SRData.x_data).detach().cpu().numpy()
            )
            # Re-assign so Individual.last_perf's @property setter fires its
            # _on_perf_change_hook (registered by PopulationManager). Without
            # this, the in-place mutations above silently bypass the hook and
            # the current_best_<version>/ mirror would not refresh. NOTE:
            # requires tmp/Individual.py merged into source_code/Individual.py;
            # if Individual.last_perf is still a plain attribute, this line is
            # a harmless no-op.
            ind.last_perf = ind.last_perf

        ind.age += SRConfig.backprop_epoch
        ind.perf_history[getattr(ind, SRConfig.ageToConsider)] = PerformanceHistory(
            performance=ind.last_perf.performance,
            nn_weights=ind.last_perf.nn_weights,
            activeNodesCoordinates=ind.last_perf.activeNodesCoordinates,
            dataTimestamp=self.data_timestamp,
        )
        log(
            f"Individual {ind.id} is not subject to tune. Its age is updated to {ind.age}. The individual will wait for individuals {self.currentlyTuned} to finish.",
            start_time=self.start_time,
            file_path=SRConfig.log_file,
            level="DEBUG",
        )
        # --- Check, if the queue contains some individual with subjectToTune==True
        self.adjust_subjecttotune_in_queue()
        # ---
        if ind.inPopulation == "mainPopulation":
            # --- self._generate_new_add_to_queue(queue)  # --- Already done in runSymReg()
            if ind.id in self.mainPopulationProcessed:
                self.mainPopulationProcessed[ind.id] += 1
            if not getattr(self, SRConfig.keepIndInMainPopulation)(ind):
                queue.popleft()
                return

        if not self.currentlyTuned:
            log(
                f"Individual {ind.id} is added to queue.",
                start_time=self.start_time,
                file_path=SRConfig.log_file,
                level="DEBUG",
            )
            queue.append(queue.popleft())
        else:
            self.waitingNotTuned[ind] = copy.deepcopy(self.currentlyTuned)
            queue.popleft()

    def _reset_main_population_processed(self):
        """
        Reset the counter of how many times an individual from main population was processed since the last merge.
        """
        self.mainPopulationProcessed = {}
        for ind in self.mainPopulation:
            self.mainPopulationProcessed[ind.id] = 0

    def _filter_queue(self, queue):
        """
        Removes
           - duplicates from the queue while preserving order,
           - individuals that are not in any population anymore.
        """
        seen = set()
        pop_ids = [ind.id for ind in self.mainPopulation]
        pop_ids.extend([ind.id for ind in self.offspring])
        queue = deque(
            ind
            for ind in queue
            if (ind.id not in seen and (seen.add(ind.id) is None)) and ind.id in pop_ids
        )
        return queue

    def _create_offspring_population(self, queue):
        """
        Recreates the offspring population by generating new individuals through crossover or mutation
        """
        self.offspring = []
        while len(self.offspring) < SRConfig.offspring_pop_size:
            new_individuals = self.crossover_mutation_reproduction(regularize=False)
            if not new_individuals:
                log(
                    f"No offspring was generated. Current offspring population size: {len(self.offspring)}.",
                    self.start_time,
                    file_path=SRConfig.log_file,
                    level="ERROR",
                )
                continue
            for new_ind in new_individuals:
                new_ind.inPopulation = "offspring"
                self.offspring.append(new_ind)
                queue.append(new_ind)
                if len(self.offspring) >= SRConfig.offspring_pop_size:
                    break
        log(
            f"Offspring population regenerated with size: {len(self.offspring)}.",
            self.start_time,
            file_path=SRConfig.log_file,
            level="HIGHLIGHT_GREEN",
        )

    def reorder_queue(self, queue):
        """
        The individuals in the queue are reordered based on their 'inPopulation' and 'backprop_iters' attributes.
        Finally, the queue is converted back to a deque.
        """
        pop_priority = {None: 0, "offspring": 1, "mainPopulation": 2}
        queue = deque(
            sorted(
                queue,
                key=lambda ind: (pop_priority[ind.inPopulation], ind.backprop_iters),
            )
        )
        log(
            f"Reordered queue ({len(queue)}): {[ind.id for ind in queue]}",
            self.start_time,
            file_path=SRConfig.log_file,
            level="HIGHLIGHT_CYAN",
        )
        log(
            f"Reordered queue ({len(queue)}): {[ind.id for ind in queue]}",
            self.start_time,
            file_path=SRConfig.filePopulation,
            level="HIGHLIGHT_CYAN",
        )
        return queue

    def adjust_subjecttotune_in_queue(self):
        """
        Adjusts the proportion of individuals with subjectToTune==True in the queue.
        """
        nbOfActive = sum(1 for ind in self.queue if ind.subjectToTune)
        k = min(3, len(self.queue))
        if nbOfActive < k:
            while True:
                ind = SRConfig.r.choice(self.queue)
                if not ind.subjectToTune:
                    ind.subjectToTune = True
                    nbOfActive += 1
                    log(
                        f"Reactivated individual in queue {ind.id}.",
                        self.start_time,
                        file_path=SRConfig.log_file,
                        level="HIGHLIGHT_GREEN",
                    )
                    if nbOfActive >= k:
                        break

    def adjust_subjecttotune_in_mainpop(self):
        """
        Adjusts the proportion of individuals with subjectToTune==True in mainPopulation.
        """
        nbOfActive = sum(1 for ind in self.mainPopulation if ind.subjectToTune)
        k = min(3, len(self.mainPopulation))
        if nbOfActive < k:
            while True:
                ind = SRConfig.r.choice(self.mainPopulation)
                if not ind.subjectToTune:
                    ind.subjectToTune = True
                    self.queue.append(ind)
                    nbOfActive += 1
                    log(
                        f"Reactivated individual in mainPopulation {ind.id}.",
                        self.start_time,
                        file_path=SRConfig.log_file,
                        level="HIGHLIGHT_GREEN",
                    )
                    if nbOfActive >= k:
                        break

    def _merge_populations(self):
        """
        TODO
        """
        log(
            "Merge of populations happens now.",
            start_time=self.start_time,
            level="HIGHLIGHT_MAGENTA",
            file_path=SRConfig.log_file,
        )
        log(
            f"Last round ended with the following main population calculations: {', '.join(f'{k} ({v})' for k, v in self.mainPopulationProcessed.items())}",
            start_time=self.start_time,
            file_path=SRConfig.log_file,
            level="DEBUG",
        )
        log(
            f"Individuals in queue ({len(self.queue)}): {[ind.id for ind in self.queue]}",
            self.start_time,
            file_path=SRConfig.log_file,
            level="DEBUG",
        )
        log(
            f"Individuals in queue ({len(self.queue)}): {[ind.id for ind in self.queue]}",
            self.start_time,
            file_path=SRConfig.filePopulation,
            level="DEBUG",
        )

        # log_population_state(
        #     population=self.mainPopulation,
        #     start_time=self.start_time,
        #     title="Merged main population",
        #     file_path=SRConfig.log_file,
        # )
        # log_population_state(
        #     population=self.mainPopulation,
        #     start_time=self.start_time,
        #     title="Merged main population",
        #     file_path=SRConfig.filePopulation,
        # )

        # log_population_state(
        #     population=self.offspring,
        #     start_time=self.start_time,
        #     title="Merged offspring population",
        #     file_path=SRConfig.log_file,
        # )
        # log_population_state(
        #     population=self.offspring,
        #     start_time=self.start_time,
        #     title="Merged offspring population",
        #     file_path=SRConfig.filePopulation,
        # )
        self.queue = self._filter_queue(self.queue)
        log(
            f"Individuals in filtered queue ({len(self.queue)}): {[ind.id for ind in self.queue]}",
            self.start_time,
            file_path=SRConfig.log_file,
            level="DEBUG",
        )

        if not SRConfig.do_merge:
            self._reset_main_population_processed()
            log(
                f"[SRConfig.do_merge==False] Next round of merge will wait until following individuals are processed: {', '.join(f'{k}, ({v})' for k, v in self.mainPopulationProcessed.items())}",
                start_time=self.start_time,
                file_path=SRConfig.log_file,
                level="DEBUG",
            )
            self.queue = self.reorder_queue(self.queue)
            return

        main_max_size = max(
            [
                ind.last_perf.performance["nbOfActiveNodes"]
                for ind in self.mainPopulation
            ]
        )
        main_orig_ids = {ind.id for ind in self.mainPopulation}
        offspring_ids = {ind.id for ind in self.offspring}

        # --------------------------------------------------------------------------------
        # --- Calculate weighted quantiles of the main population's non-dominated front
        # --------------------------------------------------------------------------------
        if SRConfig.standard_EA:
            q_valid, q_constr = self.computeQuantilesFromMaximumPerformanceValues(
                population=self.mainPopulation
            )
            log(
                f"q_valid={q_valid}, q_constr={q_constr}, main_max_size={main_max_size} calculated from mainPopulation individuals",
                start_time=self.start_time,
                file_path=SRConfig.log_file,
                level="DEBUG",
            )
            log(
                f"q_valid={q_valid}, q_constr={q_constr}, main_max_size={main_max_size} calculated from mainPopulation individuals",
                start_time=self.start_time,
                file_path=SRConfig.filePopulation,
                level="DEBUG",
            )
        else:
            # nonDominatedSet, _ = splitPopulationByDomination(pop=self.mainPopulation, front=0, criteria=SubTopology.keysAll)
            # q_valid, q_constr = self.computeQuantilesFromMaximumPerformanceValues(population=nonDominatedSet)
            q_valid, q_constr = self.computeQuantilesFromMaximumPerformanceValues(
                population=self.mainPopulation
            )
            log(
                f"q_valid={q_valid}, q_constr={q_constr}, main_max_size={main_max_size} calculated from non-dominated individuals of mainPopulation",
                start_time=self.start_time,
                file_path=SRConfig.log_file,
                level="DEBUG",
            )
            log(
                f"q_valid={q_valid}, q_constr={q_constr}, main_max_size={main_max_size} calculated from non-dominated individuals of mainPopulation",
                start_time=self.start_time,
                file_path=SRConfig.filePopulation,
                level="DEBUG",
            )

        # ------------------------------------------------------------------
        # --- Take offsprings that have at least one of  its performance values
        # --- no worse than the weighted quantiles
        # ------------------------------------------------------------------
        offspringsBelowQuantiles: list[Individual] = []
        for ind in self.offspring:
            if ind.adult and (
                ind.last_perf.performance["nbOfActiveNodes"] <= main_max_size
                or ind.last_perf.performance["valid_loss"] <= q_valid
                or ind.last_perf.performance["rmse_constr"] <= q_constr
            ):
                offspringsBelowQuantiles.append(ind)

        # ----------------------------------------------------------------------
        # --- Merge self.mainPopulation with offspringsBelowQuantiles
        # ----------------------------------------------------------------------
        res: set[Individual] = set()
        res.update(
            [
                ind
                for ind in self.mainPopulation
                if ind.last_perf.performance["nbOfActiveNodes"]
                >= SRConfig.minNbOfActiveNodes
            ]
        )
        res.update(
            [
                ind
                for ind in offspringsBelowQuantiles
                if ind.last_perf.performance["nbOfActiveNodes"]
                >= SRConfig.minNbOfActiveNodes
            ]
        )
        if len(res) > SRConfig.main_pop_size:
            self.mainPopulation = self._nds_dataset_aware(
                list(res),
                finalSize=SRConfig.main_pop_size,
                criteria=SubTopology.keysAll,  # --- SubTopology.keysPerformance
            )
        else:
            self.mainPopulation = list(res)

        main_final_ids = {ind.id for ind in self.mainPopulation}
        removed_from_main = [
            id for id in main_orig_ids if id not in main_final_ids
        ]  # --- ids removed from the main population

        # --- Remove individuals that were moved to the main population from the offspring population
        # --- and set their inPopulation attribute to 'mainPopulation'
        moved_to_main = [
            ind.id for ind in self.mainPopulation if ind.id in offspring_ids
        ]  # --- ids moved from offspring to main population
        self.offspring = [ind for ind in self.offspring if ind.id not in moved_to_main]
        for ind in self.mainPopulation:
            if ind.id in moved_to_main:
                ind.inPopulation = "mainPopulation"

        log(
            f"Individuals removed from main population: {removed_from_main}",
            start_time=self.start_time,
            file_path=SRConfig.log_file,
            level="DEBUG",
        )
        log(
            f"Individuals removed from main population: {removed_from_main}",
            start_time=self.start_time,
            file_path=SRConfig.filePopulation,
            level="DEBUG",
        )

        log(
            f"Individuals moved from offspring to main population: {moved_to_main}",
            start_time=self.start_time,
            file_path=SRConfig.log_file,
            level="DEBUG",
        )
        log(
            f"Individuals moved from offspring to main population: {moved_to_main}",
            start_time=self.start_time,
            file_path=SRConfig.filePopulation,
            level="DEBUG",
        )

        self._reset_main_population_processed()
        log(
            f"Next round of merge will wait until following individuals are processed: {', '.join(map(str, self.mainPopulationProcessed.keys()))}",
            start_time=self.start_time,
            file_path=SRConfig.log_file,
            level="DEBUG",
        )

        self.queue = self.reorder_queue(self.queue)

        return

    def perturbPopulation(self):
        """
        Perturb weights of all individuals in the main population.
        """
        log(
            f"Perturbation of population started after {self.totalBackpropsUsed} BP iterations",
            start_time=self.start_time,
            file_path=SRConfig.log_file,
            level="INFO_MAGENTA",
        )

        log_population_state(
            population=self.mainPopulation,
            start_time=self.start_time,
            title="Main Population before perturbation",
            file_path=SRConfig.filePopulation,
            tick_time=self.start_time,
            tick_backprops=self.totalBackpropsUsed,
            tick_ids=self.idCounter,
        )

        for ind in self.mainPopulation:
            st = SubTopology.createSubtopologyFromParameters(
                activeNodesCoordinates=ind.last_perf.activeNodesCoordinates,
                nnWeights=ind.last_perf.nn_weights,
                losses=ind.last_perf.performance,
            )
            st.activateAllInactiveNodes()
            _, nbOfImprovements = st.train_nn(
                learningSteps=SRConfig.totalItersNewbornPerturbed,
                learningRate=SRConfig.learning_rate_newborn,
                regularize=False,
                clipWeights=False,
                constrain=False,
                backpropItersBefore=ind.backprop_iters,
                lastPruningBackprops=ind.lastPruningBackprops,
            )
            self.numOfBackpropTunings += 1

            params = st.getSubtopologyParameters()
            st.getAnalyticFormula(simplify=False)
            outputNode = st.nnLayers[-1].units[0]
            expr = outputNode.string_analytic_expression
            simpleExpr = "NA"
            if outputNode.symbolic_expression:
                simpleExpr = outputNode.symbolic_expression.simpleExpr

            performance = {
                "valid_loss": st.losses["valid_loss"],
                "rmse_constr": st.losses["rmse_constr"],
                "complexity": st.losses["complexity"],
                "nbOfActiveNodes": st.losses["nbOfActiveNodes"],
                "nbOfImprovements": nbOfImprovements + 1,
            }
            history_update = PerformanceHistory(
                performance=performance,
                nn_weights=params["nnParams"],
                activeNodesCoordinates=params["activeNodesCoordinates"],
                dataTimestamp=self.data_timestamp,
            )
            data_output = st.get_subtopology_output(SRData.x_data)

            # Capture identity BEFORE overwriting id/backprop_iters so that
            # the perturbed individual correctly records its pre-perturbation
            # self as its parent.
            _old_id = ind.id
            _old_backprops = ind.backprop_iters

            ind.id = self.idCounter
            self.idCounter += 1

            ind.age += SRConfig.totalItersNewbornPerturbed
            ind.backprop_iters += SRConfig.totalItersNewbornPerturbed
            # Carry back the pruning watermark advanced inside the perturbation
            # train_nn (it may have pruned, since pruning is pure counter-gated).
            ind.lastPruningBackprops = st.lastPruningBackprops
            ind.adult = True
            ind.subjectToTune = True

            # Update parents and genealogy to track the perturbation event.
            ind.parents = (f"{_old_id}/{_old_backprops}",)
            _next_key = (max(ind.genealogy.keys()) + 1) if ind.genealogy else 0
            ind.genealogy = {**ind.genealogy, _next_key: ind.parents}

            ind.last_perf = history_update
            ind.perf_history = {getattr(ind, SRConfig.ageToConsider): history_update}
            ind.last_output = data_output.detach().cpu().numpy()
            ind.data_timestamp = self.data_timestamp
            ind.expr = expr
            ind.simpleExpr = simpleExpr
            ind.age_last_change_backprop = ind.age
            log(
                f"Individual {ind.id} weights perturbed.",
                self.start_time,
                file_path=SRConfig.log_file,
                level="DEBUG",
            )

        log(
            "Perturbation of population ended.",
            start_time=self.start_time,
            file_path=SRConfig.log_file,
            level="INFO_MAGENTA",
        )
        self._reset_main_population_processed()
        self.offspring = []
        self.queue = deque(self.mainPopulation)
        self.mainPopulationNotTuned = []
        self.currentlyTuned = set()
        self.waitingNotTuned = {}

        # log_population_state(
        #     population=self.mainPopulation,
        #     start_time=self.start_time,
        #     title="Main Population",
        #     file_path=SRConfig.log_file,
        # )
        # log_population_state(
        #     population=self.mainPopulation,
        #     start_time=self.start_time,
        #     title="Main Population",
        #     file_path=SRConfig.filePopulation,
        # )

        # log_population_state(
        #     population=self.offspring,
        #     start_time=self.start_time,
        #     title="Offspring Population",
        #     file_path=SRConfig.log_file,
        # )
        # log_population_state(
        #     population=self.offspring,
        #     start_time=self.start_time,
        #     title="Offspring Population",
        #     file_path=SRConfig.filePopulation,
        # )

    def updateLogFileAndSaveModelsByTime(self):
        """
        Update log files and save individual models periodically.
        """
        # Discharge any deferred current_best/ dumps so the periodic snapshot
        # and the live mirror agree on the population state at this tick.
        self.flush_pending_dumps()
        current_time = time.time()
        if current_time - self.last_log_time >= SRConfig.saveToFileTimeInterval:
            log_population_state(
                population=self.mainPopulation,
                start_time=self.start_time,
                title="Main Population",
                file_path=SRConfig.filePopulation,
            )
            log_population_state(
                population=self.offspring,
                start_time=self.start_time,
                title="Offspring Population",
                file_path=SRConfig.filePopulation,
            )
            log_population_state(
                population=self.archive,
                start_time=self.start_time,
                title="Archive",
                file_path=SRConfig.filePopulation,
            )
            log(
                f"Number of calls of backpropagation tunings: {self.numOfBackpropTunings}, Number of total backprops used: {self.totalBackpropsUsed}, Number of individuals: {self.idCounter}",
                self.start_time,
                file_path=SRConfig.filePopulation,
                level="HIGHLIGHT_INFO",
            )
            print("continues ...")
            tick = f"{int(current_time - self.start_time)}"
            for ind in self.archive:
                ind.saveIndividualPerformanceToFile(
                    folderName=f"archiveIndividuals",
                    fileName=f"archive_ind_{ind.id}",
                    tickValue=tick,
                )
                if SRConfig.humanReadableIndividualsPath.strip():
                    timestamp = f"{int(current_time - self.start_time)}"
                    filePath = (
                        SRConfig.humanReadableIndividualsPath
                        + f"time_{timestamp}/"
                        + f"status_archive_ind={ind.id}.txt"
                    )
                    ind.saveIndividualToFile(filePath=filePath)
            self._write_population_overview(
                self.archive,
                folderName="archiveIndividuals",
                tickValue=tick,
                prefix="time_",
            )
            for ind in self.mainPopulation:
                ind.saveIndividualPerformanceToFile(
                    folderName=f"mainPopulationIndividuals",
                    fileName=f"ind_{ind.id}",
                    tickValue=tick,
                )
                if SRConfig.humanReadableIndividualsPath.strip():
                    timestamp = f"{int(current_time - self.start_time)}"
                    filePath = (
                        SRConfig.humanReadableIndividualsPath
                        + f"time_{timestamp}/"
                        + f"status_mainpop_ind={ind.id}.txt"
                    )
                    ind.saveIndividualToFile(filePath=filePath)
            self._write_population_overview(
                self.mainPopulation,
                folderName="mainPopulationIndividuals",
                tickValue=tick,
                prefix="time_",
            )
            self.last_log_time = current_time

    def updateLogFileAndSaveModelsByBackprops(self):
        """
        Update log files and save individual models periodically every 'SRConfig.saveToFileBackpropsInterval' backprops.
        """
        self.flush_pending_dumps()
        if (
            self.totalBackpropsUsed - self.last_log_backprops
            >= SRConfig.saveToFileBackpropsInterval
        ):
            log_population_state(
                population=self.mainPopulation,
                start_time=self.start_time,
                title="Main Population",
                file_path=SRConfig.filePopulation,
                tick_backprops=self.totalBackpropsUsed,
            )
            log_population_state(
                population=self.offspring,
                start_time=self.start_time,
                title="Offspring Population",
                file_path=SRConfig.filePopulation,
                tick_backprops=self.totalBackpropsUsed,
            )
            log_population_state(
                population=self.archive,
                start_time=self.start_time,
                title="Archive",
                file_path=SRConfig.filePopulation,
                tick_backprops=self.totalBackpropsUsed,
            )
            log(
                f"Number of calls of backpropagation tunings: {self.numOfBackpropTunings}, Number of total backprops used: {self.totalBackpropsUsed}, Number of individuals: {self.idCounter}",
                self.start_time,
                file_path=SRConfig.filePopulation,
                level="HIGHLIGHT_INFO",
            )
            print("continues ...")
            tick = f"{int(self.totalBackpropsUsed)}"
            for ind in self.archive:
                ind.saveIndividualPerformanceToFile(
                    folderName=f"archiveIndividuals",
                    fileName=f"archive_ind_{ind.id}",
                    tickValue=tick,
                    prefix="backprops_",
                )
            self._write_population_overview(
                self.archive,
                folderName="archiveIndividuals",
                tickValue=tick,
                prefix="backprops_",
            )
            for ind in self.mainPopulation:
                ind.saveIndividualPerformanceToFile(
                    folderName=f"mainPopulationIndividuals",
                    fileName=f"ind_{ind.id}",
                    tickValue=tick,
                    prefix="backprops_",
                )
            self._write_population_overview(
                self.mainPopulation,
                folderName="mainPopulationIndividuals",
                tickValue=tick,
                    prefix="backprops_",
                )
            self.last_log_backprops = self.totalBackpropsUsed

    def updateLogFileAndSaveModelsByIds(self):
        """
        Update log files and save individual models periodically every 'SRConfig.saveToFileBackpropsInterval' backprops.
        """
        self.flush_pending_dumps()
        if self.idCounter - self.last_log_ids >= SRConfig.saveToFileIdsInterval:
            log_population_state(
                population=self.mainPopulation,
                start_time=self.start_time,
                title="Main Population",
                file_path=SRConfig.filePopulation,
                tick_ids=self.idCounter,
            )
            log_population_state(
                population=self.offspring,
                start_time=self.start_time,
                title="Offspring Population",
                file_path=SRConfig.filePopulation,
                tick_ids=self.idCounter,
            )
            log_population_state(
                population=self.archive,
                start_time=self.start_time,
                title="Archive",
                file_path=SRConfig.filePopulation,
                tick_ids=self.idCounter,
            )
            log(
                f"Number of calls of backpropagation tunings: {self.numOfBackpropTunings}, Number of total backprops used: {self.totalBackpropsUsed}, Number of individuals: {self.idCounter}",
                self.start_time,
                file_path=SRConfig.filePopulation,
                level="HIGHLIGHT_INFO",
            )
            print("continues ...")
            tick = f"{int(self.idCounter)}"
            for ind in self.archive:
                ind.saveIndividualPerformanceToFile(
                    folderName=f"archiveIndividuals",
                    fileName=f"archive_ind_{ind.id}",
                    tickValue=tick,
                    prefix="ids_",
                )
            self._write_population_overview(
                self.archive,
                folderName="archiveIndividuals",
                tickValue=tick,
                prefix="ids_",
            )
            for ind in self.mainPopulation:
                ind.saveIndividualPerformanceToFile(
                    folderName=f"mainPopulationIndividuals",
                    fileName=f"ind_{ind.id}",
                    tickValue=tick,
                    prefix="ids_",
                )
            self._write_population_overview(
                self.mainPopulation,
                folderName="mainPopulationIndividuals",
                tickValue=tick,
                    prefix="ids_",
                )
            self.last_log_ids = self.idCounter

    def runSymReg(self):
        log(
            f"Entered runSymReg method with SRConfig.client_restart_period={SRConfig.client_restart_period}.",
            self.start_time,
            file_path=SRConfig.log_file,
            level="HIGHLIGHT_GREEN",
        )

        if SRConfig.standard_EA:
            self._reset_main_population_processed()

        log(
            f"self.mainPopulation size: {len(self.mainPopulation)}",
            self.start_time,
            file_path=SRConfig.log_file,
            level="HIGHLIGHT_GREEN",
        )

        # ------------------------------------------------
        # --- Dask initialization
        # ------------------------------------------------
        self.last_log_time = time.time()
        self.last_log_backprops = 0
        self.last_log_ids = 0

        log(
            f"self.idCounter at the start of runSymReg: {self.idCounter}",
            self.start_time,
            file_path=SRConfig.log_file,
            level="HIGHLIGHT_MAGENTA",
        )

        n_cores = 4  # os.cpu_count()
        initial = deque(islice(self.mainPopulation, n_cores))
        self.queue = deque(islice(self.mainPopulation, n_cores, None))
        dask.config.set(
            {"distributed.admin.gc-diagnostics": False}
        )  # --- Turn off GC monitoring

        cluster = LocalCluster(
            n_workers=n_cores,
            threads_per_worker=1,
            processes=True,
        )
        client = Client(cluster)

        # ------------------------------------------------
        # --- Main body of the run
        # ------------------------------------------------

        # --- Parameters specific to initialPopulationLearning
        extra_args = (
            "--learning_rate",
            f"{SRConfig.learning_rate_newborn}",  # --- TODO: change to learning_rate_newborn
            "--iters",
            f"{SRConfig.backprop_epoch}",
            "--reg2rmse",
            f"{SRConfig.reg2rmseMainRun}",
            "--constr2rmse",
            f"{SRConfig.constr2rmseMainRun}",
            "--sng2rmse",
            f"{SRConfig.sng2rmse}",
        )

        futures = self.submit_initial_tasks(client, initial, specific_params=extra_args)
        future_to_key = {fut: ind_id for ind_id, fut in futures.items()}
        ac = as_completed(futures.values())
        offspring_pop_generated = False

        doNotSubmit = False
        for future in ac:

            # --- Update log files and save models periodically
            # self.updateLogFileAndSaveModelsByTime()
            self.updateLogFileAndSaveModelsByBackprops()
            # self.updateLogFileAndSaveModelsByIds()

            # --- Update max arities
            self.update_max_arities()

            key = future_to_key.pop(future, None)
            if key is not None:
                futures.pop(key, None)

            result = None
            try:
                log(
                    f"Getting future.result()",
                    self.start_time,
                    level="HIGHLIGHT_MAGENTA",
                )
                result = future.result()
            except Exception as e:
                log(f"Exception in worker: {e}", self.start_time, level="HIGHLIGHT_RED")

            # --- Always fall through to queue submission below, even if result is None or an exception occurred.
            if result is not None:
                for text in result["logs"]:
                    log(
                        f"{text}",
                        start_time=self.start_time,
                        file_path=SRConfig.log_file,
                    )

                individual, readyNotTuned = self.process_individual(result)

                # ----------------------------------------------------------
                # --- Adding individuals in queue and generating new ones
                # ----------------------------------------------------------
                if (not SRConfig.standard_EA) and individual:
                    addToQueue, _ = self._add_to_queue(individual)
                    if addToQueue:
                        self.queue.append(individual)
                        log(
                            f"Individual {individual.id} added to queue.",
                            self.start_time,
                            file_path=SRConfig.log_file,
                            level="DEBUG",
                        )

                    if (
                        all(ind.adult for ind in self.mainPopulation)
                        and individual.inPopulation == "mainPopulation"
                    ):
                        self._generate_new_add_to_queue(
                            self.queue, triggerId=individual.id
                        )

                # --- Add some offspring into mainPopulation if applicable
                if not SRConfig.standard_EA:
                    self._add_eligible_offsprings_to_main_if_not_full()

                # --- Add individuals back to queue if it should be tuned but is not
                if not SRConfig.standard_EA:
                    for ind in readyNotTuned:
                        self.queue.append(ind)
                        log(
                            f"Individual {ind.id} which is not being tuned is added back to queue.",
                            self.start_time,
                            level="DEBUG",
                        )

                # --- Merge populations if all individuals from main population were processed at least once
                if not SRConfig.standard_EA:
                    if (
                        individual
                        and self.mainPopulationProcessed.get(individual.id) == 2
                    ):
                        not_tuned_ids = [
                            ind_id
                            for ind_id, value in self.mainPopulationProcessed.items()
                            if value == 0
                        ]
                        for ind_id in not_tuned_ids:
                            log(
                                f"Individual {ind_id} was not tuned since last merge. Adding {ind_id} to queue again.",
                                self.start_time,
                                level="DEBUG",
                            )
                            tmp = next(
                                (
                                    ind
                                    for ind in self.mainPopulation
                                    if ind and ind.id == ind_id
                                ),
                                None,
                            )
                            if tmp:
                                self.queue.append(tmp)

                # --- Standard EA merge
                if SRConfig.standard_EA:
                    # --- Merge populations if all individuals from main population were processed at least once
                    if not offspring_pop_generated and (
                        all(
                            value >= 1
                            for value in self.mainPopulationProcessed.values()
                        )
                        or not self.mainPopulationProcessed
                        # any(value > 1 for value in self.mainPopulationProcessed.values())
                        or not self.mainPopulationProcessed
                    ):
                        # --- Create new offspring population
                        self.queue.clear()
                        log(
                            f"Standard EA: Creating a new offspring population.\nQueue size before self._create_offspring_population: {len(self.queue)}",
                            self.start_time,
                            level="HIGHLIGHT_MAGENTA",
                        )
                        self._create_offspring_population(self.queue)
                        log(
                            f"Queue size after self._create_offspring_population: {len(self.queue)}",
                            self.start_time,
                            level="HIGHLIGHT_MAGENTA",
                        )
                        offspring_pop_generated = True
                    if offspring_pop_generated and len(self.queue) == 0:
                        # --- Merge populations
                        log(
                            f"Standard EA: Merging populations.",
                            self.start_time,
                            level="HIGHLIGHT_MAGENTA",
                        )
                        self._merge_populations()
                        for ind in self.mainPopulation:
                            self.queue.append(ind)
                        self.offspring = []
                        offspring_pop_generated = False
                        # ---
                        self.updateArchive(self.mainPopulation)
                        self.archive.sort(
                            key=lambda ind: ind.last_perf.performance["complexity"]
                        )
                        log_population_state(
                            population=self.mainPopulation,
                            start_time=self.start_time,
                            title="Main Population",
                            file_path=SRConfig.filePopulation,
                            tick_time=self.start_time,
                            tick_backprops=self.totalBackpropsUsed,
                            tick_ids=self.idCounter,
                        )
                        log_population_state(
                            population=self.archive,
                            start_time=self.start_time,
                            title="Archive",
                            file_path=SRConfig.filePopulation,
                            tick_time=self.start_time,
                            tick_backprops=self.totalBackpropsUsed,
                            tick_ids=self.idCounter,
                        )

                # --- ANSR merge
                else:
                    if (
                        any(
                            value > 1 for value in self.mainPopulationProcessed.values()
                        )
                        or not self.mainPopulationProcessed
                    ):
                        # if all(value >= 1 for value in self.mainPopulationProcessed.values()) or not self.mainPopulationProcessed:
                        self._merge_populations()
                        self.adjust_subjecttotune_in_mainpop()
                        self.updateArchive(self.mainPopulation)
                        self.archive.sort(
                            key=lambda ind: ind.last_perf.performance["complexity"]
                        )
                        # log_population_state(
                        #     population=self.mainPopulation,
                        #     start_time=self.start_time,
                        #     title="Main Population after merge",
                        #     file_path=SRConfig.log_file,
                        #     tick_time=self.start_time,
                        #     tick_backprops=self.totalBackpropsUsed,
                        #     tick_ids=self.idCounter,
                        # )
                        # log_population_state(
                        #     population=self.offspring,
                        #     start_time=self.start_time,
                        #     title="Offspring Population after merge",
                        #     file_path=SRConfig.log_file,
                        #     tick_time=self.start_time,
                        #     tick_backprops=self.totalBackpropsUsed,
                        #     tick_ids=self.idCounter,
                        # )
                        # log_population_state(
                        #     population=self.archive,
                        #     start_time=self.start_time,
                        #     title="Archive after merge",
                        #     file_path=SRConfig.log_file,
                        #     tick_time=self.start_time,
                        #     tick_backprops=self.totalBackpropsUsed,
                        #     tick_ids=self.idCounter,
                        # )
                        # log_population_state(
                        #     population=self.mainPopulation,
                        #     start_time=self.start_time,
                        #     title="Main Population after merge",
                        #     file_path=SRConfig.filePopulation,
                        #     tick_time=self.start_time,
                        #     tick_backprops=self.totalBackpropsUsed,
                        #     tick_ids=self.idCounter,
                        # )
                        # log_population_state(
                        #     population=self.offspring,
                        #     start_time=self.start_time,
                        #     title="Offspring Population after merge",
                        #     file_path=SRConfig.filePopulation,
                        #     tick_time=self.start_time,
                        #     tick_backprops=self.totalBackpropsUsed,
                        #     tick_ids=self.idCounter,
                        # )
                        # log_population_state(
                        #     population=self.archive,
                        #     start_time=self.start_time,
                        #     title="Archive after merge",
                        #     file_path=SRConfig.filePopulation,
                        #     tick_time=self.start_time,
                        #     tick_backprops=self.totalBackpropsUsed,
                        #     tick_ids=self.idCounter,
                        # )

                # --- Hard finish
                current_time = time.time()
                if current_time - self.start_time >= SRConfig.maxTotalTime:
                    log(
                        f"Total time of {SRConfig.maxTotalTime} seconds reached. Stopping the run.",
                        start_time=self.start_time,
                        file_path=SRConfig.log_file,
                        level="HIGHLIGHT_GREEN",
                    )
                    log(
                        f"Total number of backprops used: {self.totalBackpropsUsed}.",
                        start_time=self.start_time,
                        level="HIGHLIGHT_GREEN",
                    )
                    break
                if self.totalBackpropsUsed >= SRConfig.maxTotalBackprops:
                    log(
                        f"Total number of backprops used ({self.totalBackpropsUsed}) reached. Stopping the run.",
                        start_time=self.start_time,
                        file_path=SRConfig.log_file,
                        level="HIGHLIGHT_GREEN",
                    )
                    log(
                        f"Total time: {(current_time - self.start_time):.2f} seconds.",
                        start_time=self.start_time,
                        file_path=SRConfig.log_file,
                        level="HIGHLIGHT_GREEN",
                    )
                    break

                # --- 2. client.restart() to restart the client and cluster
                # --- Perform client restart once all in-flight futures have completed
                if doNotSubmit and ac.count() == 0:
                    log(
                        f"Restarting the client ...",
                        start_time=self.start_time,
                        file_path=SRConfig.log_file,
                        level="HIGHLIGHT_MAGENTA",
                    )
                    client.restart()
                    client.wait_for_workers(n_cores)
                    log(
                        f"All workers are ready.",
                        start_time=self.start_time,
                        file_path=SRConfig.log_file,
                        level="HIGHLIGHT_MAGENTA",
                    )
                    doNotSubmit = False

            # --- Pick individual from queue
            while not doNotSubmit and self.queue:
                log(
                    f"Picked up next individual of ({self.queue[0].inPopulation}) from queue with id {self.queue[0].id}.",
                    start_time=self.start_time,
                    file_path=SRConfig.log_file,
                    level="DEBUG",
                )
                if not self.queue[0].inPopulation:
                    log(
                        f"Individual {self.queue[0].id} is not anymore in any population. Picking next individual from queue.",
                        start_time=self.start_time,
                        file_path=SRConfig.log_file,
                        level="DEBUG",
                    )
                    self.queue.popleft()
                    continue
                if not self.queue[0].subjectToTune:
                    log(
                        f"Handling subject to tune.",
                        self.start_time,
                        file_path=SRConfig.log_file,
                        level="DEBUG",
                    )
                    self._handle_not_subject_to_tune(self.queue[0], self.queue)
                    continue
                self.submit_next_task(
                    client,
                    self.queue,
                    futures,
                    ac,
                    future_to_key,
                    specific_params=extra_args,
                )
                # --- Client restart
                if self.numOfBackpropTunings % SRConfig.client_restart_period == 0:
                    doNotSubmit = True
                break
        # ---------------------------------------------------- END OF MAIN LOOP ----------------------------------------------------

        # --- Final reports
        log_population_state(
            population=self.mainPopulation,
            start_time=self.start_time,
            title="Main Population",
            file_path=SRConfig.log_file,
            tick_time=self.start_time,
            tick_backprops=self.totalBackpropsUsed,
            tick_ids=self.idCounter,
        )
        log_population_state(
            population=self.archive,
            start_time=self.start_time,
            title="Archive",
            file_path=SRConfig.log_file,
            tick_time=self.start_time,
            tick_backprops=self.totalBackpropsUsed,
            tick_ids=self.idCounter,
        )
        log(
            f"Number of calls of backpropagation tunings: {self.numOfBackpropTunings}\nNumber of total backprops used: {self.totalBackpropsUsed}, Number of individuals: {self.idCounter}",
            self.start_time,
            file_path=SRConfig.log_file,
            level="HIGHLIGHT_INFO",
        )
        print("continues ...")
        log_population_state(
            population=self.mainPopulation,
            start_time=self.start_time,
            title="Main Population",
            file_path=SRConfig.filePopulation,
            tick_time=self.start_time,
            tick_backprops=self.totalBackpropsUsed,
            tick_ids=self.idCounter,
        )
        log_population_state(
            population=self.archive,
            start_time=self.start_time,
            title="Archive",
            file_path=SRConfig.filePopulation,
            tick_time=self.start_time,
            tick_backprops=self.totalBackpropsUsed,
            tick_ids=self.idCounter,
        )
        log(
            f"Number of calls of backpropagation tunings: {self.numOfBackpropTunings}\nNumber of total backprops used: {self.totalBackpropsUsed}, Number of individuals: {self.idCounter}",
            self.start_time,
            file_path=SRConfig.filePopulation,
            level="HIGHLIGHT_INFO",
        )
        print("continues ...")

        log(
            f"self.queue size at the end of runSymReg: {len(self.queue)}",
            self.start_time,
            file_path=SRConfig.log_file,
            level="HIGHLIGHT_GREEN",
        )
        # --- Discharge any deferred current_best/ dumps so the on-disk mirror
        #     reflects the truly final population state before the run exits.
        self.flush_pending_dumps()

        # --- Save final models
        current_time = time.time()
        tick_time = f"{int(current_time - self.start_time)}"
        tick_backprops = f"{int(self.totalBackpropsUsed)}"
        tick_ids = f"{int(self.idCounter)}"
        for ind in self.archive:
            ind.saveIndividualPerformanceToFile(
                folderName=f"archiveIndividuals",
                fileName=f"archive_ind_{ind.id}",
                tickValue=tick_time,
                prefix="time_",
            )
            ind.saveIndividualPerformanceToFile(
                folderName=f"archiveIndividuals",
                fileName=f"archive_ind_{ind.id}",
                tickValue=tick_backprops,
                prefix="backprops_",
            )
            ind.saveIndividualPerformanceToFile(
                folderName=f"archiveIndividuals",
                fileName=f"archive_ind_{ind.id}",
                tickValue=tick_ids,
                prefix="ids_",
            )
        for prefix, tick in (
            ("time_", tick_time),
            ("backprops_", tick_backprops),
            ("ids_", tick_ids),
        ):
            self._write_population_overview(
                self.archive,
                folderName="archiveIndividuals",
                tickValue=tick,
                prefix=prefix,
            )
        for ind in self.mainPopulation:
            ind.saveIndividualPerformanceToFile(
                folderName=f"mainPopulationIndividuals",
                fileName=f"ind_{ind.id}",
                tickValue=tick_time,
                prefix="time_",
            )
            ind.saveIndividualPerformanceToFile(
                folderName=f"mainPopulationIndividuals",
                fileName=f"ind_{ind.id}",
                tickValue=tick_backprops,
                prefix="backprops_",
            )
            ind.saveIndividualPerformanceToFile(
                folderName=f"mainPopulationIndividuals",
                fileName=f"ind_{ind.id}",
                tickValue=tick_ids,
                prefix="ids_",
            )
        for prefix, tick in (
            ("time_", tick_time),
            ("backprops_", tick_backprops),
            ("ids_", tick_ids),
        ):
            self._write_population_overview(
                self.mainPopulation,
                folderName="mainPopulationIndividuals",
                tickValue=tick,
                prefix=prefix,
            )
        log(
            f"Final models saved",
            self.start_time,
            file_path=SRConfig.log_file,
            level="HIGHLIGHT_GREEN",
        )

        # --- Print final solution
        rmse_valid_values = [
            ind.last_perf.performance["valid_loss"] for ind in self.archive
        ]
        complexity_values = [
            ind.last_perf.performance["complexity"] for ind in self.archive
        ]
        points = [(r, c) for r, c in zip(rmse_valid_values, complexity_values)]
        # --- Choose the final model
        result: MCKPResult = select_mckp(
            points, k=2, assume_pareto_front=False, normalize_for_angles=True
        )
        best = self.archive[result.original_index]
        log(
            f"Final solution: \nvalid_loss = {best.last_perf.performance['valid_loss']}, \nrmse_constr = {best.last_perf.performance['rmse_constr']}, \ncomplexity_nn = {best.last_perf.performance['complexity']}, \nexpr = {best.expr}",
            self.start_time,
            file_path=SRConfig.log_file,
            level="HIGHLIGHT_GREEN",
            flush_file=True,
        )

        time.sleep(30)  # --- short safety delay (adjust if needed)

        # -------------------------------------------------------------
        # --- Shutdown: cancel futures, close client and cluster
        # -------------------------------------------------------------
        try:
            client.cancel(list(futures.values()), force=True)
        except Exception:
            pass
        try:
            client.close(timeout=20)
        except Exception:
            pass
        try:
            cluster.close(timeout=20)
        except Exception:
            pass
