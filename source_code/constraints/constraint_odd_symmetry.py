# Symmetry constraint
# f(...,x1,...,x2,...) = f(...,x2,...,x1,...)
import numpy as np
import copy
import warnings

warnings.filterwarnings("ignore")
import torch
from constraints.constraint import generate_xi
from SRConstraints import Constraint


def generate_samples(data, constr: Constraint):
    """
    Generates samples from data_range.
    nSamples is the total number of samples that should be equally distributed among all intervals.
    """
    labelBase = constr.name
    n = int(constr.nbOfSamples / (len(constr.domain)))  # --- samples per interval
    # ---
    newData = [
        np.array([generate_xi(lb=dimension[0], ub=dimension[1], nSamples=n, offset=d) for d, dimension in enumerate(interval)]).T
        for interval in constr.domain
    ]
    newData = np.concatenate(newData, axis=0)
    newDataSym = copy.deepcopy(newData)
    sym_var = constr.args["vars"].index(1)
    newDataSym[:, sym_var] = -1.0 * newDataSym[:, sym_var]

    # --- Original data
    data.forwardpass_data_boundaries[labelBase + "_x"] = (
        data.forwardpass_data.shape[0],
        data.forwardpass_data.shape[0] + newData.shape[0] - 1,
    )
    data.forwardpass_counts[labelBase + "_x"] = newData.shape[0]
    data.forwardpass_data = np.append(data.forwardpass_data, newData, axis=0)

    # --- Symmetric data
    data.forwardpass_data_boundaries[labelBase + "_xsym"] = (
        data.forwardpass_data.shape[0],
        data.forwardpass_data.shape[0] + newData.shape[0] - 1,
    )
    data.forwardpass_counts[labelBase + "_xsym"] = newData.shape[0]
    data.forwardpass_data = np.append(data.forwardpass_data, newDataSym, axis=0)

    return data


def update_samples(data, constr: Constraint):
    """
    TODO
    Updates constraint samples.
    nSamples is the total number of samples that should be equally distributed among all intervals.
    """
    pass


def get_constraint_term(data, constr: Constraint, y_hat, weight=1.0):
    if weight == 0.0:
        return torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    # ---
    labelBase = constr.name
    # ---
    start_x = data.forwardpass_data_boundaries[labelBase + "_x"][0]
    count_x = data.forwardpass_counts[labelBase + "_x"]
    y_hatA = y_hat[start_x : start_x + count_x]

    start_xsym = data.forwardpass_data_boundaries[labelBase + "_xsym"][0]
    count_xsym = data.forwardpass_counts[labelBase + "_xsym"]
    y_hatB = y_hat[start_xsym : start_xsym + count_xsym]

    # --- Calculate loss term
    diff = y_hatA + y_hatB  # --- Odd symmetry: f(x) = -f(-x)  ==>  diff = f(x) + f(-x)
    squared_diff = torch.square(diff)
    loss = weight * torch.sum(squared_diff) / count_x
    return loss
