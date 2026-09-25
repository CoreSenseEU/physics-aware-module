# Output on the diagonal is a half of the partial resistor
# f(x1, x2) = x1/2, for x1 == x2
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import torch
from constraints.constraint import generate_xi
from SRConstraints import Constraint


def generate_samples(data, constr:Constraint):
    """
    Generates samples from data_range.
    nSamples is the total number of samples that should be equally distributed among all intervals.
    """
    labelBase = constr.name
    n = int(constr.nbOfSamples/(len(constr.domain))) # --- samples per interval
    # ---
    d = [generate_xi(lb=interval[0][0], ub=interval[0][1], nSamples=n) for interval in constr.domain]
    d = np.concatenate(d, axis=0)
    newData = np.vstack((d,d)).T
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
    # ---
    mask = torch.from_numpy(data.forwardpass_masks[labelBase + '_x'])
    y_hatA = y_hat * mask
    y_true = (torch.from_numpy(data.forwardpass_data)[:, :1] / 2.0) * mask
    # --- Calculate loss
    diff = y_hatA - y_true
    squared_diff = torch.square(diff)
    loss = weight * torch.sum(squared_diff) / data.forwardpass_counts[labelBase + '_x']
    return loss
