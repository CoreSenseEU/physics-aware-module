import logging
import sys
import os

import SRConfig as SRConfig
from TopologyInitializer import initialize_config_and_topologies
from PopulationManager import PopulationManager


if __name__ == "__main__":
    logging.getLogger("torch").setLevel(logging.ERROR)
    cli_args = tuple(sys.argv[1:])
    masterTopology, timestamp = initialize_config_and_topologies(cli_args, readTopology=True)
    SRConfig.saveConfigurationToFile()

    os.makedirs(os.path.dirname(SRConfig.filePopulation), exist_ok=True)

    populationManager = PopulationManager(masterTopology, cli_args, timestamp)

    populationManager.runSymReg()
