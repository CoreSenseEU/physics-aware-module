import json
import numpy as np
import os, getopt
import SRConfig as SRConfig
import SRUnits as SRUnits
from logger import log


def read_params(argv):
    SRConfig.topology_file = None
    SRConfig.constraints_file = None
    outfolder = ""
    argv_str = "arguments: "
    try:
        opts, args = getopt.getopt(
            argv,
            "r:s:o:t:c:l:w:",
            [
                "seed=",
                "standard_EA=",
                "ea_ignores_stagnation=",
                "maxTotalTime=",
                "maxTotalBackprops=",
                "saveToFileBackpropsInterval=",
                "client_restart_period=",
                "outfolder=",
                "humanReadableIndividualsPath=",
                "genealogyPath=",
                "constraints=",
                "topology=",
                "init_expressions=",
                "train_data=",
                "valid_data=",
                "test_data=",
                "train_val=",
                "scale=",
                "init_pop_size=",
                "main_pop_size=",
                "offspring_pop_size=",
                "useAllLossTerms=",
                "pertIndividuals=",
                "pertBackprops=",
                "quantile=",
                "mult_coeff=",
                "min_nb_diff=",
                "rectify=",
                "minibatch=",
                "max_weights=",
                "w_thr=",
                "weight_sigma=",
                "active_units=",
                "reg_lambda=",
                "learning_rate=",
                "learning_rate_noreg=",
                "adaptive=",
                "best_ext=",
                "stack_size=",
                "update_step=",
                "sng2rmse=",
                "reg2rmse=",
                "constr2rmse=",
                "minNbOfActiveNodes=",
                "totalItersInit=",
                "iters=",
                "iters_A=",
                "reg_iters=",
                "noreg_iters=",
                "noconstr_iters=",
                "backprop_epoch=",
                "scheduler=",
                "computeTrainValidLoss=",
                "addOffspringToPopulation=",
                "addNewIndToOffspring=",
                "keepIndInMainPopulation=",
                "keepIndInOffspring=",
                "crossover=",
            ],
        )

        offspring_pop_size_changed = False
        init_pop_size_changed = False
        for opt, arg in opts:
            argv_str = argv_str + "\n" + opt.replace("-", "") + "=" + arg
            if opt == "-h":
                print(
                    "SRMain.py -s <seed> -o <outfolder> -r <reg_lambda> -t <topology> -c <constraints> -l <learning_rate>"
                    "-w <w_thr>"
                    "--weight_sigma <weight_sigma>"
                    "--learning_rate_noreg <learning_rate_noreg>"
                    "--stack_size <stack_size>",
                    "--update_step <update_step>",
                    "--main_pop_size <main_pop_size>",
                    "--offspring_pop_size <offspring_pop_size>",
                    "--quantile <quantile>",
                    "--rectify <rectify>",
                    "--mult_coeff <mult_coeff>",
                    "--min_nb_diff <min_nb_diff>",
                    "--sng2rmse <sng2rmse>",
                    "--reg2rmse <reg2rmse>",
                    "--constr2rmse <constr2rmse>",
                    "--minNbOfActiveNodes <minNbOfActiveNodes>",
                    "--adaptive <adaptive>",
                    "--best_ext <best_ext>",
                    "--max_weights <max_weights> --scale <scaleCoeff>",
                    "--totalItersInit <totalItersInit>",
                    "--iters <total_iters> --iters_A <iters_A> --reg_iters <reg_iters> --noreg_iters <noreg_iters> --noconstr_iters <noconstr_iters>",
                    "--train_val --train_data --valid_data --test_data --minibatch"
                    "--simplify <simplify>",
                    "--addOffspringToPopulation <addOffspringToPopulation>",
                    "--addNewIndToOffspring <addNewIndToOffspring>",
                    "--keepIndInMainPopulation <keepIndInMainPopulation>",
                    "--keepIndInOffspring <keepIndInOffspring>",
                    "--crossover <crossover>",
                )
                raise Exception("Printing 'help' in read_params()")
            elif opt in ("-s", "--seed"):
                SRConfig.seed = int(arg)
                SRConfig.fileMainPopulation = (
                    f"results/mainPopulation_{SRConfig.seed}.txt"
                )
                SRConfig.fileOffspring = (
                    f"results/offspringPopulation_{SRConfig.seed}.txt"
                )
            elif opt in ("--standard_EA"):
                SRConfig.standard_EA = arg.lower() == "true"
            elif opt == "--ea_ignores_stagnation":
                SRConfig.ea_ignores_stagnation = arg.lower() == "true"
            elif opt in ("-o", "--outfolder"):
                outfolder = arg
            elif opt in ("--humanReadableIndividualsPath"):
                SRConfig.humanReadableIndividualsPath = arg
            elif opt in ("--genealogyPath"):
                SRConfig.genealogyPath = arg
            elif opt in ("-t", "--topology"):
                SRConfig.topology_file = arg
            elif opt in ("-c", "--constraints"):
                SRConfig.constraints_file = arg
            elif opt in ("--maxTotalTime"):
                SRConfig.maxTotalTime = int(arg)
            elif opt in ("--maxTotalBackprops"):
                SRConfig.maxTotalBackprops = int(arg)
            elif opt in ("--saveToFileBackpropsInterval"):
                SRConfig.saveToFileBackpropsInterval = int(arg)
            elif opt in ("--client_restart_period"):
                SRConfig.client_restart_period = int(arg)
            elif opt in ("--train_val"):
                SRConfig.train_val_split = float(arg)
            elif opt in ("--init_expressions"):
                SRConfig.initPopulationFromExpressions = "init-population/" + arg
            elif opt in ("--train_data"):
                SRConfig.train_data = "data/" + arg
            elif opt in ("--valid_data"):
                SRConfig.valid_data = "data/" + arg
            elif opt in ("--test_data"):
                SRConfig.test_data = "data/" + arg
            elif opt in ("--init_pop_size"):
                SRConfig.init_pop_size = int(arg)
                init_pop_size_changed = True
            elif opt in ("--main_pop_size"):
                SRConfig.main_pop_size = int(arg)
            elif opt in ("--offspring_pop_size"):
                SRConfig.offspring_pop_size = int(arg)
                offspring_pop_size_changed = True
            elif opt in ("--pertIndividuals"):
                SRConfig.perturbationIntervalIndividuals = int(arg)
            elif opt in ("--pertBackprops"):
                SRConfig.perturbationIntervalBackprops = int(arg)
            elif opt in ("--quantile"):
                SRConfig.quantile = float(arg)
            elif opt in ("--rectify"):
                SRConfig.rectify = float(arg)
            elif opt in ("--useAllLossTerms"):
                SRConfig.useAllLossTerms = arg.lower() == "true"
            elif opt in ("--mult_coeff"):
                SRConfig.multCoeff = float(arg)
            elif opt in ("--min_nb_diff"):
                SRConfig.minNbDiffForQuantileReplacement = int(arg)
            elif opt in ("--scale"):
                SRConfig.rmseScaleCoeff = float(arg)
            elif opt in ("--stack_size"):
                SRConfig.stack_size = int(arg)
            elif opt in ("--update_step"):
                SRConfig.update_step = int(arg)
            elif opt in ("--sng2rmse"):
                SRConfig.sng2rmse = float(arg)
            elif opt in ("--reg2rmse"):
                SRConfig.reg2rmse = float(arg)
            elif opt in ("--constr2rmse"):
                SRConfig.constr2rmse = float(arg)
            elif opt in ("--minNbOfActiveNodes"):
                SRConfig.minNbOfActiveNodes = int(arg)
            elif opt in ("--adaptive"):
                if "true" in arg.lower():
                    SRConfig.adaptive = True
                elif "false" in arg.lower():
                    SRConfig.adaptive = False
                else:
                    print("Error: wrong value for 'adaptive'")
                    log(
                        "Error: wrong value for 'adaptive'",
                        start_time=SRConfig.start_time,
                        level="HIGHLIGHT_CYAN",
                    )
                    raise Exception("Something went wrong")
            elif opt in ("-l", "--learning_rate"):
                SRConfig.learning_rate = float(arg)
            elif opt in ("--weight_sigma"):
                SRConfig.weight_sigma = float(arg)
            elif opt in ("-w", "--w_thr"):
                SRConfig.activeWeightThreshold = float(arg)
            elif opt in ("--minibatch"):
                SRConfig.minibatchSize = int(arg)
            elif opt in ("--totalItersInit"):
                SRConfig.totalItersInit = int(arg)
            elif opt in ("--iters"):
                SRConfig.totalIters = int(arg)
            elif opt in ("--scheduler"):
                SRConfig.scheduler = json.loads(arg)
            elif opt in ("--addOffspringToPopulation"):
                SRConfig.addOffspringToPopulation = arg
            elif opt in ("--addNewIndToOffspring"):
                SRConfig.addNewIndToOffspring = arg
            elif opt in ("--keepIndInMainPopulation"):
                SRConfig.keepIndInMainPopulation = arg
            elif opt in ("--keepIndInOffspring"):
                SRConfig.keepIndInOffspring = arg
            elif opt in ("--crossover"):
                SRConfig.crossover = arg
            elif opt in ("--computeTrainValidLoss"):
                SRConfig.computeTrainValidLoss = arg
            elif opt == "--backprop_epoch":
                val = int(arg)
                SRConfig.backprop_epoch = val
                SRConfig.th_adult = val
                # if val in (100, 200):
                #     SRConfig.scheduler["params"]["patience"] = 50
                # elif val in (500, 1000):
                #     SRConfig.scheduler["params"]["patience"] = 200
    except getopt.GetoptError:
        print(
            "SRMain.py -s <seed> -o <outfolder> -r <reg_lambda> -t <topology> -c <constraints> -l <learning_rate>"
            "-w <w_thr>"
            "--learning_rate_noreg <learning_rate_noreg>"
            "--stack_size <stack_size>",
            "--update_step <update_step>",
            "--sng2rmse <sng2rmse>",
            "--reg2rmse <reg2rmse>",
            "--constr2rmse <constr2rmse>",
            "--adaptive <adaptive>",
            "--minNbOfActiveNodes <minNbOfActiveNodes>",
            "--best_ext <best_ext>"
            "--max_weights <max_weights> --scale <scaleCoeff>"
            "--iters <total_iters> --iters_A <iters_A> --reg_iters <reg_iters> --noreg_iters <noreg_iters> --noconstr_iters <noconstr_iters>"
            "--train_val --train_data --valid_data --test_data",
        )
        raise Exception("Something went wrong in read_params()")
    # ---
    if not init_pop_size_changed:
        SRConfig.init_pop_size = 2 * SRConfig.main_pop_size
    if not offspring_pop_size_changed:
        SRConfig.offspring_pop_size = SRConfig.main_pop_size
    SRConfig.saveToFileIdsInterval = 2 * SRConfig.main_pop_size

    argv_str = argv_str + "\n\n"
    SRConfig.outputNamePrefix = f"{outfolder}/seed={SRConfig.seed}/"
    SRConfig.filePopulation = f"{SRConfig.outputNamePrefix}population.txt"
    SRConfig.log_file = f"{SRConfig.outputNamePrefix}log.txt"
    # SRConfig.outputNamePrefix = f'{outfolder}/seed={SRConfig.seed}'
    SRConfig.outputPath = outfolder
    SRConfig.progress_file = (
        SRConfig.outputNamePrefix + "progress" + SRConfig.fileSuffix + ".txt"
    )

    if SRConfig.humanReadableIndividualsPath.strip():
        SRConfig.humanReadableIndividualsPath = (
            SRConfig.outputNamePrefix + "humanReadableIndividuals/"
        )
        os.makedirs(
            SRConfig.humanReadableIndividualsPath,
            exist_ok=True,
        )

    if SRConfig.genealogyPath.strip():
        SRConfig.genealogyPath = SRConfig.outputNamePrefix + "genealogy/"
        os.makedirs(
            SRConfig.genealogyPath,
            exist_ok=True,
        )

    np.random.seed(SRConfig.seed)
    SRUnits.init_np_seed(SRConfig.seed)
    return argv_str
