import json
import dataclasses
from typing import Any
from importlib import import_module

import SRConfig
from SRData import SRData


@dataclasses.dataclass
class Constraint:
    module: any
    name: str
    weight: float
    domain: list[list]
    nbOfSamples: int
    args: dict[str, Any]


class SRConstraints:
    constraintModules: list[Constraint] = []
    constraint_weight: float = SRConfig.constr2rmseMainRun
    static_cve_weight: float = 1.0

    def __init__(self):
        SRConstraints.constraintModules = []
        SRConstraints.constraint_weight = SRConfig.constr2rmseMainRun
        SRConstraints.static_cve_weight = 1.0

    def load_module(module_name):
        """
        It has to be in the constraints directory to be able to import.
        Parameters
        ----------
        module_name

        Returns
        -------
        Loaded module.
        """
        return import_module(f"constraints.{module_name}")

    def load_constraints(path: str):
        """
        Loads constraints from the given path.
        Parameters
           path to the file with constraints specifications
        Returns
           Constraint list containing constraint module, its weight and used variables.
        Updates
           Data structure SRData containing all data points, including the constraint samples.
        """
        if path:
            path = "./configs/" + path
        else:
            return []

        with open(path, "r") as f:
            constraints = json.load(f)["constraints"]

        SRConstraints.constraintModules = []
        constr_ids = {}
        for constraint in constraints:
            i = constr_ids.get(constraint["class"], -1)
            i += 1
            constr_ids[constraint["class"]] = i
            constr = Constraint(
                SRConstraints.load_module(constraint["class"]),
                constraint["class"] + "_" + str(i),
                constraint["weight"],
                constraint["domain"],
                constraint["nbOfSamples"],
                constraint["args"],
            )
            SRConstraints.constraintModules.append(constr)
            constr.module.generate_samples(data=SRData, constr=constr)
