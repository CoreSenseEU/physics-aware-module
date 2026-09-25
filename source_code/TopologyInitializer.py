import numpy as np
import os

import SRConfig
from SRConstraints import SRConstraints
from SRData import SRData
from utils import NNTopology
from MasterTopology import MasterTopology
from SRConfigParser import read_params


def initialize_config_and_topologies(
    argv, seed: int = None, readTopology: bool = False, start_time: float = None
):
    read_params(argv)

    if not seed: seed = SRConfig.seed
    SRConfig.r = np.random.default_rng(seed=seed)
    if start_time is not None: SRConfig.start_time = start_time
    SRConfig.saveConfigurationToFile()
    # SRConfig.totalIters = SRConfig.totalItersMainRun
    # SRConfig.reg2rmse = SRConfig.reg2rmseMainRun

    # --- Init data
    timeStamp = os.path.getmtime(SRConfig.train_data)
    SRData.init_data(train_file=SRConfig.train_data, val_file=SRConfig.valid_data, test_file=SRConfig.test_data)

    # --- Init constraints
    SRConstraints.load_constraints(path=SRConfig.constraints_file)

    if readTopology:
        masterTopology = read_topology()
        return masterTopology, timeStamp
    else:
        return None, timeStamp


def read_topology():

    NNTopology.all_layer_defs, NNTopology.all_identities = (
        NNTopology.get_topology_definitions(topology_file=SRConfig.topology_file)
    )
    for defs, ids in zip(NNTopology.all_layer_defs, NNTopology.all_identities):
        masterTopology = MasterTopology(
            inputSize=SRData.x_data.shape[1], layer_defs=defs, identities=ids
        )
        return masterTopology

    return None
