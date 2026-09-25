import numpy as np
import copy
import sympy as sp


def round_constants_sympy(e):
    ndigits = 3
    return e.replace(
        lambda x: x.is_Number and not x.is_Integer,
        lambda x: sp.Float(round(float(x), ndigits)),
    )


def compare_expressions(
    expr1: str,
    expr2: str,
    ranges: list,
    n_samples: int = 1000,
    seed: int = 42,
    simplify_timeout: float = 10.0,
) -> float:
    """
    Computes the RMSE between two analytical expressions evaluated on
    randomly sampled input data.

    expr1 is first simplified using sympy.simplify() with a timeout so
    that constant folding is applied before evaluation. If simplification
    times out or fails for any reason, the original expr1 is used unchanged.

    Parameters
    ----------
    expr1            : str   -- first expression, e.g. "0.5*np.tanh(x0) + 3.0*x1"
                                (uses np. prefix for numpy functions)
    expr2            : str   -- second expression, e.g. "0.5*np.tanh(x0) + 3.0*x1"
                                (may use plain tanh/sin without np. prefix)
    ranges           : list  -- list of [min, max] per input variable,
                                e.g. [[-1, 1], [-5, 5]] for x0, x1
    n_samples        : int   -- number of random samples (default 1000)
    seed             : int   -- random seed for reproducibility (default 42)
    simplify_timeout : float -- max seconds allowed for sympy.simplify() on expr1
                                (default 10.0). If exceeded, original expr1 is used.

    Returns
    -------
    float -- RMSE between expr1 and expr2 over the sampled inputs
    """
    from utils_timeout import run_with_timeout_sympy

    # --- Numpy function names that need np. prefix for eval()
    np_functions = ("tanh", "sin", "cos", "exp", "log", "sqrt", "abs")

    try:
        # -------------------------------------------------------------------
        # --- Simplify expr1 using sympy.simplify() with timeout.
        # --- Remove np. prefix before passing to sympy, restore it after.
        # -------------------------------------------------------------------
        expr1_orig = copy.deepcopy(expr1)
        for fn in np_functions:
            expr1_orig = expr1_orig.replace(f"np.{fn}(", f"{fn}(")

        result_orig, is_timeout, error_message = run_with_timeout_sympy(
            expr_str=expr1_orig,
            timeout_seconds=simplify_timeout,
            method="simplify",
        )

        if result_orig is not None and not is_timeout:
            print(f"\nOriginal expr1:         {expr1_orig}")
            print(f"Simplified expr1:       {result_orig}")
            # --- Restore np. prefix for eval()
            for fn in np_functions:
                result_orig = result_orig.replace(f"{fn}(", f"np.{fn}(")
                expr1_orig = expr1_orig.replace(f"{fn}(", f"np.{fn}(")

        # -------------------------------------------------------------------
        # --- Round constants in original expr1 and simplify again
        # -------------------------------------------------------------------
        expr1_orig_round = copy.deepcopy(expr1)
        expr1_orig_round = expr1_orig_round.replace("np.", "")
        expr1_orig_round = sp.sympify(expr1_orig_round)
        expr1_orig_round = round_constants_sympy(expr1_orig_round)
        result_orig_round, is_timeout, error_message = run_with_timeout_sympy(
            expr_str=str(expr1_orig_round),
            timeout_seconds=simplify_timeout,
            method="simplify",
        )

        if result_orig_round is not None and not is_timeout:
            print(f"Original expr1_round:   {expr1_orig_round}")
            print(f"Simplified expr1_round: {result_orig_round}")
            # --- Restore np. prefix for eval()
            for fn in np_functions:
                result_orig_round = result_orig_round.replace(f"{fn}(", f"np.{fn}(")

        print(f"expr2:                  {expr2}\n")

    except Exception:
        pass

    # --- Restore np. prefix in expr2 in case it came from sympy output
    # --- (sympy outputs plain tanh/sin without np. prefix)
    for fn in np_functions:
        expr2 = expr2.replace(f"{fn}(", f"np.{fn}(")

    rng = np.random.default_rng(seed=seed)
    n_vars = len(ranges)

    # --- 1) Sample random uniform test data
    samples = np.column_stack(
        [rng.uniform(low=r[0], high=r[1], size=n_samples) for r in ranges]
    )  # shape: (n_samples, n_vars)

    # --- Build evaluation namespace: x0, x1, x2, ... + numpy
    namespace = {"np": np}
    for i in range(n_vars):
        namespace[f"x{i}"] = samples[:, i]

    # -------------------------------------------------------
    # --- 2) Evaluate all expressions
    # -------------------------------------------------------
    try:
        y1_orig = np.array(eval(expr1_orig, namespace), dtype=float)
    except Exception as e:
        raise ValueError(f"Failed to evaluate expr1_orig: '{expr1_orig}'\nError: {e}")

    try:
        y1_simple = np.array(eval(result_orig, namespace), dtype=float)
    except Exception as e:
        raise ValueError(f"Failed to evaluate expr1: '{result_orig}'\nError: {e}")

    try:
        y1_round_simple = np.array(eval(result_orig_round, namespace), dtype=float)
    except Exception as e:
        raise ValueError(
            f"Failed to evaluate result_orig_round: '{result_orig_round}'\nError: {e}"
        )

    try:
        y2 = np.array(eval(expr2, namespace), dtype=float)
    except Exception as e:
        raise ValueError(f"Failed to evaluate expr2: '{expr2}'\nError: {e}")

    # --- 3) Compute RMSE
    rmse_orig_simple = float(np.sqrt(np.mean((y1_orig - y1_simple) ** 2)))
    rmse_orig_round_simple = float(np.sqrt(np.mean((y1_orig - y1_round_simple) ** 2)))
    rmse_exp1_exp2 = float(np.sqrt(np.mean((y1_orig - y2) ** 2)))

    return rmse_orig_simple, rmse_orig_round_simple, rmse_exp1_exp2


# =========================================================

if __name__ == "__main__":
    expr1: str = "0.965182708*np.sin(1.4120422345*x0)+0.4547085015*x1"
    expr2: str = "0.458*x1 + 0.968*sin(1.383*x0)"

    expr1: str = (
        "0.9989979947*np.sin(1.4987928023*x0)+0.4682192351*((0.9991733098*x1) * (0.9991733098*x1))+0.018247321*x1"
    )
    expr2: str = "0.462*x1**2 + 0.021*x1 + 0.999*sin(1.5*x0)"

    # expr1: str = (
    #     "0.0425827879*np.sin(0.1227145469*np.sin(1.4985120778*x0))+0.9936646437*np.sin(1.4985120778*x0)+0.4685081145*((0.9959471297*x1) * (0.9959471297*x1))+0.0202694817*x1"
    # )
    # expr2: str = (
    #     "0.466*x1**2 + 0.019*x1 + 0.999*sin(1.499*x0) + 0.087*sin(0.221*sin(1.499*x0))"
    # )

    ranges = [[-1, 1], [-1, 1]]
    rmse_orig_simple, rmse_orig_round_simple, rmse_exp1_exp2 = compare_expressions(
        expr1=expr1,
        expr2=expr2,
        ranges=ranges,
        n_samples=100,
        seed=11,
        simplify_timeout=2.0,
    )
    print(f"RMSE(expr1_orig, expr1_simple): {rmse_orig_simple:.6f}")
    print(f"RMSE(expr1_orig, expr1_round_simple): {rmse_orig_round_simple:.6f}")
    print(f"RMSE(expr1_orig, expr2): {rmse_exp1_exp2:.6f}")
