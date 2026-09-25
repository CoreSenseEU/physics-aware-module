# Monotonically decreasing with a positive second derivative w.r.t. the variable k
# f(x_c) >= f(x_r),
# f(x_l) >= f(x_c),
# f(x_c) - f(x_r) < f(x_l) - f(x_c)
# for x_l = x_c - eps*v and x_r = x_c + eps*v,
# where v[i] == 0 for i != k and v[i] == 1 for i == k
# and k is the index of the variable for which the constraint is checked
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import torch
import copy
from constraints.constraint import generate_xi
from SRConstraints import Constraint


def generate_samples(data, constr:Constraint):
    """
    Generates samples from data_range.
    nSamples is the total number of samples that should be equally distributed among all intervals.
    """
    labelBase = constr.name
    eps = constr.args['eps']
    varId = constr.args['var']
    n = int(constr.nbOfSamples/(len(constr.domain))) # --- samples per interval
    # ---
    data_c = [np.array([generate_xi(lb=d[0]+eps, ub=d[1]-eps, nSamples=n) for d in i]).T for i in constr.domain]
    data_c = np.concatenate(data_c, axis=0)
    # ---
    data_l = copy.deepcopy(data_c)
    data_l[:, varId] = data_l[:, varId] - eps
    # ---
    data_r = copy.deepcopy(data_c)
    data_r[:, varId] = data_r[:, varId] + eps
    # ---
    data.forwardpass_data_boundaries[labelBase+'_c'] = (data.forwardpass_data.shape[0], data.forwardpass_data.shape[0] + data_c.shape[0] - 1)
    data.forwardpass_counts[labelBase+'_c'] = data_c.shape[0]
    data.forwardpass_data = np.append(data.forwardpass_data, data_c, axis=0)
    data.forwardpass_data_boundaries[labelBase+'_l'] = (data.forwardpass_data.shape[0], data.forwardpass_data.shape[0] + data_l.shape[0] - 1)
    data.forwardpass_counts[labelBase+'_l'] = data_l.shape[0]
    data.forwardpass_data = np.append(data.forwardpass_data, data_l, axis=0)
    data.forwardpass_data_boundaries[labelBase+'_r'] = (data.forwardpass_data.shape[0], data.forwardpass_data.shape[0] + data_r.shape[0] - 1)
    data.forwardpass_counts[labelBase+'_r'] = data_r.shape[0]
    data.forwardpass_data = np.append(data.forwardpass_data, data_r, axis=0)
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
    start_c = data.forwardpass_data_boundaries[labelBase + '_c'][0]
    count_c = data.forwardpass_counts[labelBase + '_c']
    y_hat_c = y_hat[start_c:start_c + count_c]

    start_l = data.forwardpass_data_boundaries[labelBase + '_l'][0]
    count_l = data.forwardpass_counts[labelBase + '_l']
    y_hat_l = y_hat[start_l:start_l + count_l]

    start_r = data.forwardpass_data_boundaries[labelBase + '_r'][0]
    count_r = data.forwardpass_counts[labelBase + '_r']
    y_hat_r = y_hat[start_r:start_r + count_r]
    # --- Calculate loss
    diff_rc = y_hat_r - y_hat_c
    loss_rc = torch.sum(torch.square(torch.maximum(diff_rc, torch.tensor(0.)))) / count_c

    diff_cl = y_hat_c - y_hat_l
    loss_cl = torch.sum(torch.square(torch.maximum(diff_cl, torch.tensor(0.)))) / count_c

    diff_cr = y_hat_c - y_hat_r
    diff_lc = y_hat_l - y_hat_c
    diff_cr_lc = diff_cr - diff_lc
    loss_cr_lc = torch.sum(torch.square(torch.maximum(diff_cr_lc, torch.tensor(0.)))) / count_c

    loss = weight * (loss_rc + loss_cl + loss_cr_lc)
    return loss
