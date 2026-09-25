import torch
import numpy as np
import math
import os
import time

import sympy as sp
import traceback

import SRConfig
from SRData import SRData
from SubTopology import SubTopology
from TopologyInitializer import initialize_config_and_topologies
from Individual import PerformanceHistory

from prefix import infix_to_prefix, count_operators
from utils_timeout import run_with_timeout_sympy


def run_parallel_finetune(
    cli_args: tuple,
    activeNodesCoordinates,
    nn_weights,
    performance,
    indId,
    indPopulation,
    start_time,
    backpropIters=0,
    lastPruningBackprops=0,
):
    # --- Restrict PyTorch to 1 CPU thread per worker (prevents oversubscription)
    torch.set_num_threads(1)
    try:
        finetuner = ParallelFinetuning()
        result = finetuner.parallelFineTuneSubtopologies(
            cli_args,
            activeNodesCoordinates,
            nn_weights,
            performance,
            indId,
            indPopulation,
            start_time=start_time,
            backpropIters=backpropIters,
            lastPruningBackprops=lastPruningBackprops,
        )
        return result
    except SystemExit as e:
        traceback.print_exc()
        return None  # --- return cleanly instead of dying
    except BaseException as e:
        traceback.print_exc()
        return None


class ParallelFinetuning:
    seed = 1
    r = np.random.default_rng(seed=seed)
    masterTopology = None

    @staticmethod
    def initData(argv, seed: int, start_time: float):
        masterTopology, timeStamp = initialize_config_and_topologies(
            argv, seed=seed, readTopology=True, start_time=start_time
        )
        ParallelFinetuning.masterTopology = masterTopology
        return timeStamp

    @staticmethod
    def simplify_subtopology(
        masterTopology, subTopology: SubTopology, exprToSubtopology: bool = True
    ):

        def round_constants_sympy(e, ndigits=SRConfig.simplificationRoundDecimals):
            return e.replace(
                lambda x: x.is_Number and not x.is_Integer,
                lambda x: sp.Float(round(float(x), ndigits)),
            )

        expr = subTopology.getAnalyticFormula(simplify=False)
        expr = expr.replace("np.", "")
        expr = sp.sympify(expr)

        if SRConfig.simplificationRoundConstants and exprToSubtopology:
            expr = round_constants_sympy(expr)

        result, is_timeout, error_message = run_with_timeout_sympy(
            str(expr),
            timeout_seconds=SRConfig.simplificationTimeout,
            method=SRConfig.simplificationMethod,
        )

        expr = result if (result is not None and not is_timeout) else None
        simplified_expr = "NA" if expr is None else expr

        if not exprToSubtopology:
            return None, simplified_expr

        # --- Continue only if exprToSubtopology == True.
        # --- Otherwise, we just want the simplified expression string without converting back to subtopology
        if not expr:
            return None, "NA"

        pref_expr = infix_to_prefix(expr, masterTopology.availableOperators)
        op_num = count_operators(pref_expr)
        if op_num > SRConfig.maxOpsInExpr:
            return None, "NA"

        try:
            simplified: SubTopology = masterTopology.ExprToSubTopology(pref_expr)
            if simplified is None:
                return None, "NA"
            # -------------------------------
            # --- Update simplified.losses
            # -------------------------------
            simplified.losses = subTopology.losses.copy()  # --- deep copy
            simplified.updateActiveNodes()
            return simplified, simplified_expr

        except Exception as e:
            return None, "NA"

    def parallelFineTuneSubtopologies(
        self,
        argv,
        activeNodesCoordinates,
        nn_weights,
        performance,
        indId,
        indPopulation,
        start_time: float,
        backpropIters: int = 0,
        lastPruningBackprops: int = 0,
    ):
        # --- Load training data into this Dask worker and verify the file's mtime
        #     did not move during the read (i.e. no writer touched it mid-read,
        #     which would yield a torn snapshot). Retry with a short sleep if it
        #     did, up to MAX_ATTEMPTS, then accept the last read with a warning.
        #
        #     Exact == on mtimes (both sides come from os.path.getmtime so they
        #     are bit-identical when the file is unchanged) - the previous
        #     math.isclose(..., rel_tol=1e-9, abs_tol=1e-9) had an effective
        #     tolerance of ~1.78 s at Unix-timestamp magnitudes, so any file
        #     change finishing within ~2 s of the worker's read was silently
        #     accepted as "stable" and could result in training on a torn read.
        MAX_ATTEMPTS = 20      # 20 * 0.1 s = up to ~2 s of waiting for the writer
        RETRY_SLEEP_S = 0.1
        timestamp = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            timestamp = ParallelFinetuning.initData(
                argv, seed=indId, start_time=start_time
            )
            if timestamp == os.path.getmtime(SRConfig.train_data):
                if attempt > 1:
                    print(
                        f"[ParallelFinetuning worker indId={indId}] training data "
                        f"settled after {attempt}/{MAX_ATTEMPTS} attempts "
                        f"(mtime={timestamp})"
                    )
                break
            time.sleep(RETRY_SLEEP_S)
        else:
            # for/else: ran without break -> never converged
            print(
                f"WARNING [ParallelFinetuning worker indId={indId}] training data "
                f"did not stabilize after {MAX_ATTEMPTS} attempts; accepting last "
                f"read (read mtime={timestamp}, "
                f"current mtime={os.path.getmtime(SRConfig.train_data)})"
            )
        subTopology = SubTopology.createSubtopologyFromParameters(
            activeNodesCoordinates=activeNodesCoordinates,
            nnWeights=nn_weights,
            losses=performance,
            text=f"parallelFineTuneSubtopologies-indId={indId}",
        )
        logs = []

        # ---------------------------------------------------------------------
        # ----- Structural simplification (topology rewrite from sympy)
        # ----- This is done BEFORE backprop so we train on the simpler structure.
        # ---------------------------------------------------------------------
        simplified_expr = "NA"
        if performance["nbOfActiveNodes"] <= SRConfig.maxNodesToSimplify:
            try:
                simplified, _ = self.simplify_subtopology(
                    self.masterTopology, subTopology, indId, start_time, logs
                )
                if simplified:
                    subTopology = simplified
            except BaseException as e:
                pass

        if (
            SRConfig.timeToApplyLasso == "before"
            and subTopology.nnLayers[-1].units[0].name == "UnitIdentLS"
        ):
            subTopology.applyLeastSquaresAndUpdate(logs, indId, start_time)

        regularize = SRConfig.reg2rmse > 0.0
        constrain = SRConfig.constr2rmse > 0.0

        _, nbOfImprovements = subTopology.train_nn(
            learningSteps=SRConfig.totalIters,
            learningRate=SRConfig.learning_rate,
            regularize=regularize,
            constrain=constrain,
            clipWeights=True,
            logs=logs,
            backpropItersBefore=backpropIters,
            lastPruningBackprops=lastPruningBackprops,
        )

        if (
            SRConfig.timeToApplyLasso == "after"
            and subTopology.nnLayers[-1].units[0].name == "UnitIdentLS"
        ):
            subTopology.applyLeastSquaresAndUpdate(logs, indId, start_time)

        params = subTopology.getSubtopologyParameters()

        # --- Generate expression and simple expression
        subTopology.getAnalyticFormula(simplify=False)
        outputNode = subTopology.nnLayers[-1].units[0]
        expr = outputNode.string_analytic_expression

        # ---------------------------------------------------------------------
        # ----- Simplification
        # ---------------------------------------------------------------------
        simplified_expr = "NA"
        if subTopology.losses["nbOfActiveNodes"] <= SRConfig.maxNodesToSimplify:
            try:
                _, simplified_expr = self.simplify_subtopology(
                    self.masterTopology, subTopology, exprToSubtopology=False
                )
            except BaseException as e:
                simplified_expr = "NA"

        performance = {
            "valid_loss": subTopology.losses["valid_loss"],
            "rmse_constr": subTopology.losses["rmse_constr"],
            "complexity": subTopology.losses["complexity"],
            "nbOfActiveNodes": subTopology.losses["nbOfActiveNodes"],
            "nbOfImprovements": nbOfImprovements,
        }
        history_update = PerformanceHistory(
            performance=performance,
            nn_weights=params["nnParams"],
            activeNodesCoordinates=params["activeNodesCoordinates"],
            dataTimestamp=timestamp,
            # formula=expr if simplified_expr == "NA" else simplified_expr
        )
        data_output = subTopology.get_subtopology_output(SRData.x_data)

        return {
            "id": indId,
            "iters": SRConfig.totalIters,
            "historyUpdate": history_update,
            "output": data_output,
            "logs": logs,
            "expr": expr,
            "simpleExpr": simplified_expr,
            "lastPruningBackprops": subTopology.lastPruningBackprops,
        }
