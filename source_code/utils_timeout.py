import multiprocessing
import time
import tempfile
import os
import json
import uuid
import subprocess
import sys


def run_with_timeout_sympy(expr_str, timeout_seconds=10, method="factor"):
    result_path = os.path.join(
        tempfile.gettempdir(), f"sympy_result_{uuid.uuid4().hex}.json"
    )
    expr_path = os.path.join(
        tempfile.gettempdir(), f"sympy_expr_{uuid.uuid4().hex}.json"
    )

    # --- Write expression to file
    with open(expr_path, "w", encoding="utf-8") as f:
        json.dump({"expr": expr_str}, f)

    # --- Launch a fresh Python interpreter directly via subprocess.Popen
    # --- This does NOT use multiprocessing and avoids the DLL loader lock
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sympy_worker.py")
    p = subprocess.Popen(
        [sys.executable, script, expr_path, result_path, method],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if p.poll() is not None:
            break
        time.sleep(0.05)

    if p.poll() is None:
        p.kill()
        p.wait()
        try:
            os.unlink(expr_path)
            os.unlink(result_path)
        except OSError:
            pass
        return None, True, "Timed out"

    try:
        os.unlink(expr_path)
    except OSError:
        pass

    if p.returncode != 0:
        try:
            os.unlink(result_path)
        except OSError:
            pass
        return None, False, f"Worker exited with code {p.returncode}"

    if not os.path.exists(result_path):
        return None, False, "No result file written"

    try:
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return None, False, f"Failed to read result: {e}"
    finally:
        try:
            os.unlink(result_path)
        except OSError:
            pass

    if data["status"] == "err":
        return None, False, data["error"]

    return data["result"], False, None


def _worker_fn_file(func, args, kwargs, result_path):
    try:
        result = func(*args, **kwargs)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump({"status": "ok", "result": result}, f)
    except BaseException as e:
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump({"status": "err", "error": str(e)}, f)


def run_with_timeout(func, args=(), kwargs=None, timeout_seconds=10):
    if kwargs is None:
        kwargs = {}

    fd, result_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.unlink(result_path)

    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(
        target=_worker_fn_file,
        args=(func, args, kwargs, result_path),
        daemon=True,
    )
    p.start()

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if not p.is_alive():
            break
        time.sleep(0.05)

    if p.is_alive():
        p.kill()
        p.join()
        try:
            os.unlink(result_path)
        except OSError:
            pass
        return None, True, "Timed out"

    if p.exitcode != 0:
        try:
            os.unlink(result_path)
        except OSError:
            pass
        return None, False, f"Worker process exited with code {p.exitcode}"

    if not os.path.exists(result_path):
        return None, False, "No result file written by worker process"

    try:
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return None, False, f"Failed to read result file: {e}"
    finally:
        try:
            os.unlink(result_path)
        except OSError:
            pass

    if data["status"] == "err":
        return None, False, data["error"]

    return data["result"], False, None


if __name__ == "__main__":
    import inspect
    from distributed import LocalCluster

    print(inspect.signature(LocalCluster.__init__))

    from distributed import LocalCluster, Worker

    cluster = LocalCluster(
        n_workers=4,
        threads_per_worker=1,
        processes=True,
    )

    import multiprocessing
    import os

    print(
        f"[WORKER] is_daemon={multiprocessing.current_process().daemon}, pid={os.getpid()}",
        flush=True,
    )
