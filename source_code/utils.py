import sys
import SRConfig
from logger import log
import re
from pathlib import Path
from sympy import (
    symbols,
    sympify,
    simplify,
    Function,
    Mul,
    Add,
    Pow,
)
from sympy.parsing.sympy_parser import parse_expr
from call_function_with_timeout import SetTimeout
from collections import defaultdict

from SRUnits import (
    UnitMultiply,
    UnitMultiply_b,
    UnitDivide,
    UnitDivide_b,
    UnitReciprocal,
    UnitSin,
    UnitSin_b,
    UnitCos,
    UnitCos_b,
    UnitTanh,
    UnitTanh_b,
    UnitArcTan,
    UnitArcTan_b,
    UnitSquare,
    UnitSquare_b,
    UnitSqrt,
    UnitSqrt_b,
    UnitCube,
    UnitCube_b,
    UnitSign,
    UnitIdent,
    UnitIdent1,
    UnitIdentLS,
)


def parse_individual_file(filepath: Path):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    valid_loss_match = re.search(r"%\s*valid_loss:\s*([\d.eE+-]+)", content)
    rmse_constr_match = re.search(r"%\s*rmse_constr:\s*([\d.eE+-]+)", content)
    complexity_match = re.search(r"%\s*complexity:\s*([\d.eE+-]+)", content)
    m = re.search(r"^\s*output\s*=\s*(?P<formula>.*?);\s*$", content, re.MULTILINE)
    expr = m.group("formula")[1:-1] if m else None

    valid_loss = float(valid_loss_match.group(1)) if valid_loss_match else None
    rmse_constr = float(rmse_constr_match.group(1)) if rmse_constr_match else None
    complexity = float(complexity_match.group(1)) if complexity_match else None

    return valid_loss, rmse_constr, complexity, expr


def matlab_to_python(expr: str) -> str:
    """Convert Matlab syntax to Python."""
    expr = expr.replace(".^", "**")
    expr = expr.replace("^", "**")
    expr = expr.replace(".*", "*")
    expr = expr.replace("./", "/")
    return expr


def simplify_expression_to_python(
    orig_expr: str, simplification_method=None, timeout=15
):
    """
    Expects the expression in a raw str format, neither NumPy- nor Matlab-like things.
    """
    x1, x2, x3 = symbols("x1 x2 x3")

    if simplification_method is None:
        simplification_method = simplify

    # --- Removes '1.0*' when it's not part of a larger number
    orig_expr = re.sub(r"\b1\.0\*", "", orig_expr)
    expr = sympify(orig_expr)

    simplify_with_timeout = SetTimeout(simplification_method, timeout=timeout)
    is_done, is_timeout, error_message, expr = simplify_with_timeout(expr)
    expr = str(expr) if is_done else orig_expr

    return expr


def count_ops_with_breakdown(expr_str: str):
    expr = parse_expr(expr_str, evaluate=False)

    counts = defaultdict(int)

    def visit(node):
        # --- Function call (e.g., sin, tanh)
        if isinstance(node, Function):
            counts[f"func:{node.func.__name__}"] += 1
        # --- Addition
        elif isinstance(node, Add):
            counts["add"] += len(node.args) - 1
        # --- Multiplication
        elif isinstance(node, Mul):
            args = node.args
            # Skip unary minus (e.g., -1 * x)
            args_filtered = [arg for arg in args if not (arg.is_Number and arg == -1)]
            counts["mul"] += len(args_filtered) - 1
        # --- Power
        elif isinstance(node, Pow):
            counts["pow"] += 1

        for arg in node.args:
            visit(arg)

    visit(expr)
    return sum(counts.values()), dict(counts)


def python_to_matlab(expr: str) -> str:
    """Convert Python syntax to Matlab."""
    expr = expr.replace("**", ".^")
    expr = expr.replace("*", ".*")
    expr = expr.replace("/", "./")
    expr = expr.replace("tanh", "tanh")
    expr = expr.replace("sin", "sin")
    expr = expr.replace("cos", "cos")
    expr = expr.replace("sign", "sign")
    return expr


class NNTopology:
    unit_types = {
        "UnitMultiply": UnitMultiply,
        "UnitMultiply_b": UnitMultiply_b,
        "UnitDivide": UnitDivide,
        "UnitDivide_b": UnitDivide_b,
        "UnitReciprocal": UnitReciprocal,
        "UnitSin": UnitSin,
        "UnitSin_b": UnitSin_b,
        "UnitCos": UnitCos,
        "UnitCos_b": UnitCos_b,
        "UnitTanh": UnitTanh,
        "UnitTanh_b": UnitTanh_b,
        "UnitArcTan": UnitArcTan,
        "UnitArcTan_b": UnitArcTan_b,
        "UnitSquare": UnitSquare,
        "UnitSquare_b": UnitSquare_b,
        "UnitSqrt": UnitSqrt,
        "UnitSqrt_b": UnitSqrt_b,
        "UnitCube": UnitCube,
        "UnitCube_b": UnitCube_b,
        "UnitSign": UnitSign,
        "UnitIdent": UnitIdent,
        "UnitIdentLS": UnitIdentLS,
        "UnitIdent1": UnitIdent1,
    }
    all_layer_defs = None
    all_identities = None

    def __init__(self):
        pass

    def get_topology_definitions(topology_file: str = ""):
        """
        Reads possibly multiple topology definitions from a file.
        :param topology_file:
        :return: lists of layer_defs and identities
        """
        all_layer_defs = []
        all_identities = []
        nn_layer_defs = []
        nn_identities = []
        if topology_file is not None:
            with open("topologies/" + topology_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if "topology" in line:
                        if nn_layer_defs:
                            all_layer_defs.append(nn_layer_defs)
                            all_identities.append(nn_identities)
                        # ---
                        nn_layer_defs = []
                        nn_identities = []
                        continue
                    if line.startswith("Unit"):
                        l = {}  # --- new layer
                        line = line.split(",")
                        for u in line:
                            u = u.split(":")
                            l[NNTopology.unit_types[u[0].strip()]] = int(u[1])
                        nn_layer_defs.append(l)
                    elif line.startswith("identities"):
                        line = line.split(":")
                        if line[1].strip().lower() == "true":
                            nn_identities.append(True)
                        elif line[1].strip().lower() == "false":
                            nn_identities.append(False)
                        else:
                            log(
                                "get_topology_definitions(): Wrong identities value!",
                                start_time=SRConfig.start_time,
                                level="HIGHLIGHT_CYAN",
                            )
                            return None, None
                    else:
                        log(
                            "get_topology_definitions(): Wrong topology file content!",
                            start_time=SRConfig.start_time,
                            level="HIGHLIGHT_CYAN",
                        )
                        return None, None
                if nn_layer_defs:
                    all_layer_defs.append(nn_layer_defs)
                    all_identities.append(nn_identities)
        return all_layer_defs, all_identities


# --------------------------------------------------------------------------
# nnTopology = NNTopology()
