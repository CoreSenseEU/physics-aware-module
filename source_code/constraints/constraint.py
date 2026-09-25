# Common methods for constraints
import sys
import numpy as np

import SRConfig


def generate_xi(lb, ub, nSamples, boundaries=True, offset=0):
    """
    Generates nSamples random values from the interval <lb, ub>.
    The set of generated samples includes the boundary values.
    """
    # x = lb + (ub - lb) * SRConfig.r.random(size=nSamples)  # --- Original
    r_static = np.random.default_rng(SRConfig.seed + offset)
    x = lb + (ub - lb) * r_static.random(size=nSamples)

    if boundaries and nSamples > 1:
        # boundaryPos = SRConfig.r.permutation(x=nSamples)  # --- Original
        boundaryPos = r_static.permutation(x=nSamples)
        x[boundaryPos[0]] = lb
        x[boundaryPos[1]] = ub
    return x


def eliminate_forbidden_samples(data, domain, forbidden):
    """
        Eliminates all samples with forbidden structure
    :param data:
    :param forbidden:
    :return:
    """
    forbidden = forbidden.split(";")
    forbidden_patterns = np.zeros(shape=[len(forbidden), data.shape[1]])

    for k, p in enumerate(forbidden):
        forbidden_dims = p.replace("[", "").replace("]", "").split(",")
        if len(forbidden_dims) < data.shape[1]:
            print(
                f'constraint_exactvalue.args["forbidden"] has an invalid shape',
                file=sys.stderr,
            )
            exit(1)
        for i, el in enumerate(forbidden_dims):
            try:
                float(el)
                forbidden_patterns[k, i] = float(el)
            except ValueError:
                forbidden_patterns[k, i] = np.inf  # --- any finite value is OK
    # --- Purify the data
    for p in forbidden_patterns:
        for i, r in enumerate(data):
            test_p = np.array(
                [el_p if not np.isinf(el_p) else el_r for el_p, el_r in zip(p, r)]
            )
            if np.all(r == test_p):
                # --- Regenerate illegal sample
                interval = SRConfig.r.integers(0, len(domain))
                x_new = np.array(
                    [
                        generate_xi(
                            lb=dimension[0],
                            ub=dimension[1],
                            nSamples=1,
                            boundaries=False,
                        )
                        for dimension in domain[interval]
                    ]
                ).T
                data[i] = x_new
    # ---
    return data
