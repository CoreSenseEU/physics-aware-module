import random
import os
import time

# -----------------------------------------------
# Files
# -----------------------------------------------
topology_file: str
constraints_file: str
outputPath: str
outputNamePrefix: str
progress_file: str
humanReadableIndividualsPath: str = (
    ""  # --- path to save individuals in a human-readable format
)
genealogyPath: str = ""  # --- path to save individuals for genealogy analysis

# --- Enable/disable the on-disk mirror of mainPopulation and archive into
#     <outputNamePrefix><folder>/current_best/. Updated on every mutation of
#     either population or whenever any tracked individual's last_perf changes.
#     Throttled so bursts coalesce into one dump.
saveCurrentBest: bool = True

# --- Minimum seconds between two current_best/ dumps of the same population.
#     A mutation or perf change that arrives within this window of the previous
#     dump only marks the population dirty; the deferred write lands at the
#     next call to PopulationManager.flush_pending_dumps() (which is invoked
#     from each updateLogFileAndSaveModelsBy* and once at end of runSymReg).
#     Set to 0.0 to dump on every change; raise for less I/O.
saveCurrentBestMinIntervalS: float = 5.0

train_data: str
valid_data: str = None
test_data: str = None

# -----------------------------------------------
# Functions
# -----------------------------------------------
selection_1 = (
    "tournamentSelectionFront"  # --- tournamentSelectionRMSE, tournamentSelectionFront
)
selection_2 = "tournamentSelectionFront"  # --- tournamentSelectionRMSE, tournamentSelectionFront, tournamentSelectionUniqueGenealogy, tournamentSelectionUniqueGenealogyB
crossover = "subTopologyCrossoverB"  # --- subTopology_crossover_limited_arity | crossover_limited_arity_topology_mixing | subTopologyCrossoverB
addNewIndToOffspring = (
    "addNewIndToOffspring"  # --- addNewIndToOffspring / addNewIndToOffspringIfDominates
)
addOffspringToPopulation = "addOffspringToPopulationNever"  # --- addOffspringToPopulation / addOffspringToPopulationNever
keepIndInMainPopulation = "keepIndInMainPopulationAlways"  # --- keepIndInMainPopulationAlways / keepIndInMainPopulationConditional
keepIndInOffspring = "keepIndInOffspringConditional"  # --- keepIndInOffspringAlways / keepIndInOffspringConditional
networkActivityMutation = "networkActivityMutation"  # --- networkActivityMutation / networkActivityMutationOnlyToInactive
computeTrainValidLoss = "get_RMSE"

# -----------------------------------------------
# Scheduler
# -----------------------------------------------
scheduler = {
    "name": "ReduceLROnPlateau",
    "params": {
        "mode": "min",
        "factor": 0.5,  # --- 0.25
        "patience": 50,  # --- 50
        "cooldown": 0,
        "min_lr": 1e-6,
    },
}

# --- maximum number of active input weights for specific unit types (if not specified, the default is 1000)
max_arities: dict[str, int] = {}
arity_schedules: dict[str, int] = {}

# -----------------------------------------------
# Variables
# -----------------------------------------------
seed: int = 1
precisionDigits: int = 10
filePopulation: str = None
log_file: str = None
r: random = None
standard_EA: bool = (
    False  # --- True ... standard EA, False ... EA with open-ended refinement of models
)

# --- initPopulationFromDirectory: path to a folder with txt files, where each file contains a low-level definition of a Subtopology
# --- that are to be used to seed the initial population of models
initPopulationFromDirectory: str = ""
# --- initPopulationFromExpressions: path to the txt file with analytical expressions that are to be used to seed
# --- the initial population of models
# --- ex.: "init-population/init_drone.txt", "init-population/init_drone_sihao_2026_01_07_a.txt", "init_resistors.txt"
initPopulationFromExpressions: str = ""

start_time: float = time.time()
maxTotalTime: int = 36000  # --- maximum total time of the run in seconds
maxTotalBackprops: int = (
    50000  # --- maximum number of backpropagation iterations over the whole run
)
client_restart_period: int = 5000  # --- number of tuning epochs between client restarts
archiveIdMin: int = (
    1000000  # --- minimum allowed ID of individuals saved in the archive
)

saveToFileTimeInterval: int = (
    120  # --- save the population information to file every 'saveToFileTimeInterval' seconds
)
saveToFileBackpropsInterval: int = (
    5000  # --- save the population information to file every 'saveToFileBackpropsInterval' backprops used
)
saveToFileIdsInterval: int = (
    20  # --- save the population information to file every 'saveToFileIdsInterval' individuals generated
)

lassoAlpha: float = 1.0  # --- Lasso regularization coefficient
timeToApplyLasso: str = (
    "never"  # --- before/after/never: apply Lasso regularization before/after train_nn, or do not apply it at all
)

init_pop_size: int = 20
main_pop_size: int = 10
offspring_pop_size: int = 10
do_merge: bool = True

tournamentSize1: int = 2
tournamentSize2: int = 2

ageToConsider: str = (
    "backprop_iters"  # --- backprop_iters/age used in comparison of individuals
)
replaceDominatedBy: str = (
    "similarity"  # --- similarity/size when there are more dominated individuals remove either the most similar or the biggest one
)

quantile: float = (
    1.0  # --- quantile used when we try to add offspring individual to the main population
)
multCoeff: float = 1.2
minNbDiffForQuantileReplacement: int = (
    3  # --- minimum difference in the number of active nodes in the main pop for quantile-based replacement
)

# ------------------------------------------------------
# ------- Simplification parameters
# ------------------------------------------------------
simplificationMethod: str = "simplify"  # --- "cancel", "simplify"
simplificationWeightThreshold: float = (
    0.01  # --- if the weight is below this threshold, whole subtree is removed during simplification
)
maxNodesToSimplify: int = (
    7  # --- maximum number of nodes in the subTopology to be simplified
)
simplificationTimeout: int = 4  # --- timeout for the simplification process in seconds
simplificationRoundConstants: bool = (
    True  # --- whether to round constants in the simplified expression
)
simplificationRoundDecimals: int = (
    3  # --- number of decimals to round constants in the simplified expression
)
maxOpsInExpr: int = (
    10  # --- maximum number of operators in the expression to be mapped to subtopology
)

# ------------------------------------------------------
th_adult: int = (
    100  # --- number of backpropagation iterations after which the subTopology is considered an adult
)
generated_set_size: int = (
    1  # --- number of candidate subtopologies generated for each crossover/mutation action
)
nbOfNewbornIterations: int = (
    100  # --- number of backprop iterations used to tune newly generated subtopologies
)
backprop_epoch: int = (
    100  # --- number of backpropagation iterations after which new individuals are generated
)
max_backprop_iters_offspring: int = (
    500  # --- maximum number of backpropagation iterations for an offspring
)
max_noimprovment_epochs_main: int = (
    5  # --- set 'SubjectToTune' to False after 'max_noimprovment_epochs' successive backprop_epochs without improvement
)
ea_ignores_stagnation: bool = (
    True  # --- True ... in standard EA, ignore stagnation for mainpop individuals
)
max_noimprovment_epochs_offspring: int = (
    2  # --- set 'SubjectToTune' to False after 'max_noimprovment_epochs' successive backprop_epochs without improvement
)
softDomination: bool = (
    True  # --- use soft domination in train_nn if true, otherwise use hard domination only
)
useAllLossTerms: bool = (
    True  # --- True ... use all available loss terms in train_nn(), False ... use a subset of loss terms
)

initActiveUnitsProp = 1.0
keysAll = ["valid_loss", "rmse_constr", "nbOfActiveNodes"]

useCommonParents: bool = (
    False  # --- True ... use common parents for all crossover/mutation actions within 'generated_set_size'
)
rectify: float = 0.5  # --- probability of using the subTopology rectification operator

crossoverRate: float = 0.8
mutationAfterCrossoverRate: float = 0.2
applyActivityOrConfigMutation: float = 0.50
inheritWeights: float = 0.80
nodeActivityMutationRate: float = 0.25  # --- nodeActivityMutationRate: 0.1
nodeConfigMutationRate: float = 0.1
nodeUndersampledMutationRate: float = 0.1
nodeOversampledMutationRate: float = 0.1
utilLow: float = 0.4  # --- 0.4
utilHigh: float = 0.6  # --- 0.6

# --- 0 < x < 1 ... only a subset of parameters is updated.
perturbation: float = 0.5
perturbationIntervalIndividuals = (
    0  # --- number of newly generated individuals between two perturbations
)
perturbationIntervalBackprops = (
    0  # --- number of backprops calculated between two perturbations
)
totalItersNewbornPerturbed: int = 100

totalIters: int
totalItersInit: int = (
    500  # --- 0 ... just to check what happens if there is no learning involved
)
finalIters: int = 0
alternatePrefs: bool = True

# --------------------------------------------------------------------
# ------- Adaptive constraints/regularization weights paramaters
# --------------------------------------------------------------------
reg2rmse: float = 0.0  # --- current reg2rmse
reg2rmseMainRun: float = 0.5
constr2rmse: float = 0.0  # --- current constr2rmse
constr2rmseMainRun: float = 0.5
sng2rmse: float = 0.5
# ---
adaptive: bool = True
stack_size: int = 10
update_step: int = 1
scaleCoeff: float = 1.0
rmseScaleCoeff: float = 1.0

reg_type: str = "L05"
train_val_split: float = 0.8
learning_rate_newborn: float = 0.025  # --- 0.025
learning_rate: float = 0.01  # --- current learning rate 0.01
learning_subset_size: float = (
    1.0  # --- x == 1.0 ... all subTopology parameters are updated in each backprop iteration.
)

# --------------------------------------------------------------------
# ------- Pruning parameters
# --------------------------------------------------------------------
pruning: bool = False
pruningStep: int = 200  # --- number of backprops between prunings
pruningIters: int = 50  # --- number of iterations applied to the pruned subTopology
pruningOutputRelTol: float = (
    0.01  # --- data-relative unit-output cutoff: a unit is inactivated if its maximum
    #     absolute output across all samples is below pruningOutputRelTol * std(y_train).
    #     Dimensionless (fraction of the target spread), so it is scale-invariant.
)
pruningMinWeightRatio: float = (
    0.01  # --- if the absolute value of the weight is less than 'pruningMinWeightRatio * maximumWeight', then the weight is pruned
)

# --------------------------------------------------------------------
# ------- Weights management paramaters
# --------------------------------------------------------------------
activeWeightThreshold: float = 0.0001  # --- 0.0001
weight_min_init: float = 0.05
weight_max_init: float = 0.5  # --- 0.5
weight_sigma: float = 0.2  # --- 0.2
weight_init_one: float = 1.0  # --- probability that the weight is initialized to +/-1
# ---
minNbOfActiveNodes: int = 2
maxFrontNb: int = 1000


# --------------------------------------------------------------------
# ------- Saving solutions to file paramaters
# --------------------------------------------------------------------
saveBestTopologiesStep: int = 5
savePickleFinal: bool = False
savePickleIntermediate: bool = False
fileSuffix: str = ""


# =========================================================================================
def saveConfigurationToFile():
    os.makedirs(os.path.dirname(f"{outputNamePrefix}configuration.txt"), exist_ok=True)
    f = open(outputNamePrefix + f"configuration" + fileSuffix + ".txt", "w")

    f.write(f"seed: {seed}\n")
    f.write(f"totalTime: {maxTotalTime}\n")
    f.write(f"maxTotalBackpropIters: {maxTotalBackprops}\n")
    f.write(f"client_restart_period: {client_restart_period}\n")
    f.write(f"standard_EA: {standard_EA}\n")

    f.write(f"\n")
    f.write(f"saveToFileTimeInterval: {saveToFileTimeInterval}\n")
    f.write(f"saveToFileBackpropsInterval: {saveToFileBackpropsInterval}\n")
    f.write(f"saveToFileIdsInterval: {saveToFileIdsInterval}\n")

    f.write(f"\n")
    f.write(f"selection_1: {selection_1}\n")
    f.write(f"tournamentSize1: {tournamentSize1}\n")
    f.write(f"selection_2: {selection_2}\n")
    f.write(f"tournamentSize2: {tournamentSize2}\n")
    f.write(f"crossover: {crossover}\n")
    f.write(f"addNewIndToOffspring: {addNewIndToOffspring}\n")
    f.write(f"addOffspringToPopulation: {addOffspringToPopulation}\n")
    f.write(f"keepIndInMainPopulation: {keepIndInMainPopulation}\n")
    f.write(f"keepIndInOffspring: {keepIndInOffspring}\n")
    f.write(f"networkActivityMutation: {networkActivityMutation}\n")
    f.write(f"computeTrainValidLoss: {computeTrainValidLoss}\n")
    f.write(f"scheduler: {scheduler}\n")

    f.write(f"\n")
    f.write(f"init_pop_size: {init_pop_size}\n")
    f.write(f"main_pop_size: {main_pop_size}\n")
    f.write(f"offspring_pop_size: {offspring_pop_size}\n")
    f.write(f"ageToConsider: {ageToConsider}\n")
    f.write(f"replaceDominatedBy: {replaceDominatedBy}\n")
    f.write(f"initActiveUnitsProp: {initActiveUnitsProp}\n")

    f.write(f"\n")
    f.write(f"quantile: {quantile}\n")
    f.write(f"multCoeff: {multCoeff}\n")
    f.write(f"minNbDiffForQuantileReplacement: {minNbDiffForQuantileReplacement}\n")
    f.write(f"keysAll: {keysAll}\n")

    f.write(f"\n")
    f.write(f"th_adult: {th_adult}\n")
    f.write(f"backprop_epoch: {backprop_epoch}\n")
    f.write(f"max_backprop_iters_offspring: {max_backprop_iters_offspring}\n")
    f.write(f"generated_set_size: {generated_set_size}\n")
    f.write(f"nbOfNewbornIterations: {nbOfNewbornIterations}\n")
    f.write(f"max_noimprovment_epochs_main: {max_noimprovment_epochs_main}\n")
    f.write(f"ea_ignores_stagnation: {ea_ignores_stagnation}\n")
    f.write(f"max_noimprovment_epochs_offspring: {max_noimprovment_epochs_offspring}\n")
    f.write(f"totalItersInit: {totalItersInit}\n")
    f.write(f"finalIters: {finalIters}\n")
    f.write(f"alternatePrefs: {alternatePrefs}\n")
    f.write(f"finalIters: {finalIters}\n")
    f.write(f"softDomination: {softDomination}\n")
    f.write(f"useAllLossTerms: {useAllLossTerms}\n")

    f.write(f"\n")
    f.write(f"useCommonParents: {useCommonParents}\n")
    f.write(f"rectify: {rectify}\n")

    f.write(f"\n")
    f.write(f"crossoverRate: {crossoverRate}\n")
    f.write(f"mutationAfterCrossoverRate: {mutationAfterCrossoverRate}\n")
    f.write(f"applyActivityOrConfigMutation: {applyActivityOrConfigMutation}\n")
    f.write(f"nodeActivityMutationRate: {nodeActivityMutationRate}\n")
    f.write(f"nodeConfigMutationRate: {nodeConfigMutationRate}\n")
    f.write(f"nodeUndersampledMutationRate: {nodeUndersampledMutationRate}\n")
    f.write(f"nodeOversampledMutationRate: {nodeOversampledMutationRate}\n")
    f.write(f"utilLow: {utilLow}\n")
    f.write(f"utilHigh: {utilHigh}\n")
    f.write(f"inheritWeights: {inheritWeights}\n")

    f.write(f"\n")
    f.write(f"reg2rmse: {reg2rmse}\n")
    f.write(f"reg2rmseMainRun: {reg2rmseMainRun}\n")
    f.write(f"constr2rmse: {constr2rmse}\n")
    f.write(f"constr2rmseMainRun: {constr2rmseMainRun}\n")
    f.write(f"sng2rmse: {sng2rmse}\n")

    f.write(f"\n")
    f.write(f"maxOpsInExpr: {maxOpsInExpr}\n")
    f.write(f"simplificationMethod: {simplificationMethod}\n")
    f.write(f"simplificationWeightThreshold: {simplificationWeightThreshold}\n")
    f.write(f"simplificationTimeout: {simplificationTimeout}\n")
    f.write(f"simplificationRoundConstants: {simplificationRoundConstants}\n")
    f.write(f"simplificationRoundDecimals: {simplificationRoundDecimals}\n")
    f.write(f"maxNodesToSimplify: {maxNodesToSimplify}\n")

    f.write(f"\n")
    f.write(f"activeWeightThreshold: {activeWeightThreshold}\n")
    f.write(f"weight_min_init: {weight_min_init}\n")
    f.write(f"weight_max_init: {weight_max_init}\n")
    f.write(f"weight_sigma: {weight_sigma}\n")
    f.write(f"weight_init_one: {weight_init_one}\n")

    f.write(f"\n")
    f.write(f"reg_type: {reg_type}\n")
    f.write(f"train_val_split: {train_val_split}\n")
    f.write(f"learning_rate_newborn: {learning_rate_newborn}\n")
    f.write(f"learning_subset_size: {learning_subset_size}\n")
    f.write(f"\n")
    f.write(f"perturbation: {perturbation}\n")
    f.write(f"perturbationInterval: {perturbationIntervalIndividuals}\n")
    f.write(f"perturbationIntervalBackprops: {perturbationIntervalBackprops}\n")
    f.write(f"totalItersNewbornPerturbed: {totalItersNewbornPerturbed}\n")

    f.write(f"\n")
    f.write(f"pruningStep: {pruningStep}\n")
    f.write(f"pruningIters: {pruningIters}\n")
    f.write(f"pruningOutputRelTol: {pruningOutputRelTol}\n")
    f.write(f"pruningMinWeightRatio: {pruningMinWeightRatio}\n")

    f.write(f"\n")
    f.write(f"adaptive: {adaptive}\n")
    f.write(f"stack_size: {stack_size}\n")
    f.write(f"update_step: {update_step}\n")
    f.write(f"rmseScaleCoeff: {rmseScaleCoeff}\n")

    f.write(f"\n")
    f.write(f"minNbOfActiveNodes: {minNbOfActiveNodes}\n")

    f.write(f"\n")
    f.write(f"topology_file: {topology_file}\n")
    f.write(f"constraints_file: {constraints_file}\n")
    f.write(f"outputPath: {outputPath}\n")
    f.write(f"outputNamePrefix: {outputNamePrefix}\n")
    f.write(f"progress_file: {progress_file}\n")
    f.write(f"humanReadableIndividualsPath: {humanReadableIndividualsPath}\n")
    f.write(f"saveCurrentBest: {saveCurrentBest}\n")
    f.write(f"saveCurrentBestMinIntervalS: {saveCurrentBestMinIntervalS}\n")
    f.write(f"\n")
    f.write(f"train_data: {train_data}\n")
    f.write(f"valid_data: {valid_data}\n")
    f.write(f"test_data: {test_data}\n")
    f.write(f"\n")
    f.write(f"initPopulationFromDirectory: {initPopulationFromDirectory}\n")
    f.write(f"initPopulationFromExpressions: {initPopulationFromExpressions}\n")
    f.write(f"savePickleFinal: {savePickleFinal}\n")
    f.write(f"savePickleIntermediate: {savePickleIntermediate}\n")
    f.write(f"fileSuffix: {fileSuffix}\n")
    f.close()
