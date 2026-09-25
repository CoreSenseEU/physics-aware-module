# --- Output is close to the specified value 'v' with the accepted tolerance 'eps'.
# --- abs(f(X) - v) <= eps
import sys
import numpy as np
import warnings

warnings.filterwarnings("ignore")
import torch
from constraints.constraint import generate_xi, eliminate_forbidden_samples
from SRConstraints import Constraint


def generate_samples(data, constr: Constraint):
    """
    Generates constraint samples.
    nSamples is the total number of samples that should be equally distributed among all intervals.
    """
    labelBase = constr.name
    n = int(constr.nbOfSamples / (len(constr.domain)))  # --- samples per interval

    # --- Check the type of the argument "value"
    v = constr.args["value"]
    try:
        float(v)
    except ValueError:
        print(f'constraint_close_to_value.args["value"] must be a scalar', file=sys.stderr)
        exit(1)
    eps = constr.args["eps"]
    try:
        float(eps)
    except ValueError:
        print(f'constraint_close_to_value.args["eps"] must be a scalar', file=sys.stderr)
        exit(1)

    # --- Generate constraint samples
    newData = [np.array([generate_xi(lb=dimension[0], ub=dimension[1], nSamples=n) for dimension in interval]).T for interval in constr.domain]
    newData = np.concatenate(newData, axis=0)

    # --- Eliminate all samples with forbidden structure
    if "forbidden" in constr.args.keys():
        forbidden = constr.args["forbidden"].strip()
        newData = eliminate_forbidden_samples(newData, constr.domain, forbidden)

    # --- Add the generated samples to the data structures
    data.forwardpass_data_boundaries[labelBase + "_x"] = (data.forwardpass_data.shape[0], data.forwardpass_data.shape[0] + newData.shape[0] - 1)
    data.forwardpass_counts[labelBase + "_x"] = newData.shape[0]
    data.forwardpass_data = np.append(data.forwardpass_data, newData, axis=0)
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
    value = constr.args["value"]
    eps = constr.args["eps"]
    # ---
    if not isinstance(y_hat, torch.Tensor):
        y_hat = torch.tensor(y_hat, dtype=torch.float64, requires_grad=True)

    mask = data.forwardpass_masks[labelBase + "_x"]
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask, dtype=torch.float64)

    counts = data.forwardpass_counts[labelBase + "_x"]
    if not isinstance(counts, torch.Tensor):
        counts = torch.tensor(counts, dtype=torch.float64)

    y_hatA = y_hat * mask

    # --- Calculate loss for the constraint: abs(f(X) - value) <= eps
    # --- Two-sided hinge: no penalty while the output stays within the
    # --- tolerance band [value - eps, value + eps]; quadratic penalty on the
    # --- amount by which abs(f(X) - value) exceeds eps. This generalizes the
    # --- one-sided hinge of constraint_gtvalue / constraint_ltvalue (which use
    # --- torch.maximum(..., 0)) to both directions via torch.abs, on top of the
    # --- squared-error structure of constraint_exactvalue.
    # --- NOTE: like its sibling value-constraints, this relies on value == 0 so
    # --- that mask-zeroed rows (y_hatA == 0) yield abs(0 - value) - eps = -eps,
    # --- which clamps to 0 and does not contribute to the sum.
    diff = torch.abs(y_hatA - value) - eps
    positive_diff = torch.maximum(diff, torch.tensor(0.0, dtype=torch.float64))
    squared_diff = torch.square(positive_diff)
    mean_squared_diff = torch.sum(squared_diff) / counts
    loss = weight * torch.sqrt(mean_squared_diff)

    return loss
