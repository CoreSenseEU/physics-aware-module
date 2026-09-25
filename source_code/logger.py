import os
import time
from colorama import Fore, Back, Style, init

init(autoreset=True)


def align_text(message: str, start_time: float = None) -> str:
    timestamp = f"[{time.time() - start_time:7.2f}] " if start_time else ""
    indent = " " * len(timestamp)

    lines = message.strip().splitlines()
    formatted_lines = [f"{timestamp}{lines[0]}"] + [
        f"{indent}{line}" for line in lines[1:]
    ]

    full_msg = "\n".join(formatted_lines)
    return full_msg


def log(
    message: str,
    start_time: float = None,
    file_path=None,
    level="INFO",
    flush_file: bool = False,
):
    """Prints aligned multi-line log messages either to file or standard output."""

    COLORS = {
        "INFO": Fore.BLUE,
        "HIGHLIGHT_INFO": Back.GREEN + Fore.WHITE,
        "INFO_MAGENTA": Back.LIGHTMAGENTA_EX + Fore.BLACK,
        "WARNING": Back.YELLOW + Fore.BLACK,
        "HIGHLIGHT_GREEN": Back.GREEN + Fore.BLACK,
        "HIGHLIGHT_RED": Back.RED + Fore.BLACK,
        "HIGHLIGHT_CYAN": Back.LIGHTCYAN_EX + Fore.BLACK,
        "HIGHLIGHT_MAGENTA": Back.LIGHTMAGENTA_EX + Fore.BLACK,
        "DEBUG": Back.RED + Fore.BLACK,
        "ERROR": Back.RED + Fore.BLACK,
    }

    if "INFO" in level.upper():
        # if True:
        full_msg = align_text(message, start_time)
        print(f"{COLORS.get(level.upper(), '')}{full_msg}{Style.RESET_ALL}", flush=True)

        if file_path:
            with open(file_path, "a") as f:
                f.write(full_msg + "\n")
                if flush_file:
                    f.flush()
                    os.fsync(f.fileno())


def log_population_state(
    population,
    start_time,
    title,
    file_path=None,
    tick_time=None,
    tick_backprops=None,
    tick_ids=None,
):
    """Logs the current state of a population."""
    header_line = f"{title}:"
    header_line += (
        f" tick_time={time.time() - tick_time:7.2f}," if tick_time is not None else ""
    )
    header_line += (
        f" tick_backprops={tick_backprops}," if tick_backprops is not None else ""
    )
    header_line += f" tick_ids={tick_ids}" if tick_ids is not None else ""
    # Local import to avoid a top-of-module circular import (Individual.py
    # imports from logger.py too via PopulationManager.py indirectly).
    from Individual import latest_perf

    lines = [header_line]
    for i, ind in enumerate(population):
        # Defensive: an individual whose perf_history was just wiped by
        # PopulationManager._advance_dataset_version has no entries yet,
        # so fall back to last_perf via latest_perf(). If even that is None
        # (never-measured individual), write 'n/a' placeholders rather than
        # crash the logger - this method is called from many run-state
        # snapshots and must not break the EA.
        perf = latest_perf(ind)
        if perf is not None:
            valid_loss_s = round(perf.performance["valid_loss"], 9)
            rmse_constr_s = round(perf.performance["rmse_constr"], 9)
            complexity_s = perf.performance["complexity"]
        else:
            valid_loss_s = "n/a"
            rmse_constr_s = "n/a"
            complexity_s = "n/a"
        lines.append(
            f"{i}.\t id: {ind.id}, age: {ind.age}, backprop_iters: {ind.backprop_iters}, "
            f"valid_loss: {valid_loss_s}, "
            f"rmse_constr: {rmse_constr_s}, "
            f"complexity: {complexity_s}, "
            f"subjectToTune: {ind.subjectToTune}"
        )
    lines.append("   ")

    message = "\n".join(lines)
    log(message, start_time, file_path=file_path, level="INFO")
