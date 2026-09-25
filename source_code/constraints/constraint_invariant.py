# Output is equal to the specified state variable x_i
# f(x) == x_i
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import torch
from constraints.constraint import generate_xi
from SRConstraints import Constraint


def generate_samples(data, constr:Constraint):
    """
    Generates samples from the constraint's domain.
    nSamples is the total number of samples that should be equally distributed among all domain's intervals.
    """
    labelBase = constr.name
    n = int(constr.nbOfSamples / (len(constr.domain))) # --- samples per interval
    # ---
    newData = [np.array([generate_xi(lb=dimension[0], ub=dimension[1], nSamples=n) for dimension in interval]).T for interval in constr.domain]
    newData = np.concatenate(newData, axis=0)
    # ---
    data.forwardpass_data_boundaries[labelBase+'_x'] = (data.forwardpass_data.shape[0], data.forwardpass_data.shape[0] + newData.shape[0] - 1)
    data.forwardpass_counts[labelBase+'_x'] = newData.shape[0]
    data.forwardpass_data = np.append(data.forwardpass_data, newData, axis=0)
    return data


def update_samples(data, constr:Constraint):
    """
    TODO
    Updates constraint samples.
    nSamples is the total number of samples that should be equally distributed among all intervals.
    """
    pass


def get_constraint_term(data, constr:Constraint, y_hat, weight=1.0):
    if weight == 0.0:
        return torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    # ---
    labelBase = constr.name
    varId = constr.args['var']
    # ---
    var_values = data.forwardpass_data[:, varId].reshape(-1, 1) * data.forwardpass_masks[labelBase+'_x']
    y_hatA = y_hat * data.forwardpass_masks[labelBase + '_x']
    # --- Calculate loss
    diff = var_values - y_hatA
    squared_diff = torch.square(diff)
    loss = weight * torch.sum(squared_diff) / data.forwardpass_counts[labelBase + '_x']
    return loss
