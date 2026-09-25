# sympy_worker.py
# Minimal module with NO heavy imports (no torch, no numpy, no SRConfig etc.)
# This is the only module the spawned subprocess needs to import.
import json
import sys


def simplify_from_string(expr_str, result_path, method="factor"):
    try:
        import sympy
        from sympy import sympify

        expr = sympify(expr_str)

        sympy_methods = {
            "simplify": sympy.simplify,
            "factor": sympy.factor,
            "cancel": sympy.cancel,
        }
        if method not in sympy_methods:
            raise ValueError(
                f"Unknown simplification method: '{method}'. Choose from: {list(sympy_methods.keys())}"
            )

        result = str(sympy_methods[method](expr))
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump({"status": "ok", "result": result}, f)
    except BaseException as e:
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump({"status": "err", "error": str(e)}, f)


if __name__ == "__main__":
    expr_path = sys.argv[1]
    result_path = sys.argv[2]
    method = sys.argv[3] if len(sys.argv) > 3 else "factor"
    with open(expr_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    simplify_from_string(data["expr"], result_path, method=method)
