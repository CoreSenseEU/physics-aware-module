from __future__ import annotations

from dataclasses import dataclass
from math import atan2
from typing import Sequence
from pathlib import Path

# import matplotlib.pyplot as plt
import numpy as np

from utils import (
    parse_individual_file,
    matlab_to_python,
    simplify_expression_to_python,
    count_ops_with_breakdown,
    python_to_matlab,
)


@dataclass(frozen=True)
class MCKPResult:
    point: tuple[float, float]  # --- (rmse, complexity)
    original_index: int  # --- index in the input sequence
    front_points: np.ndarray  # --- Pareto front, sorted by complexity
    front_input_indices: np.ndarray  # --- input indices of front_points
    knee_front_indices: np.ndarray  # --- candidate knee indices within front_points
    interior_front_indices: np.ndarray  # --- front indices that got bend angles
    bend_angles: np.ndarray  # --- bend angles for interior_front_indices

    angle_cluster_labels: np.ndarray | None = None  # --- one label per interior point
    angle_cluster_centers: np.ndarray | None = None
    selected_cluster: int | None = None


def _pareto_front_minimize(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.lexsort((points[:, 0], points[:, 1]))
    sorted_points = points[order]

    keep_sorted_positions = []
    best_rmse_so_far = np.inf

    for pos, (rmse, complexity) in enumerate(sorted_points):
        if rmse < best_rmse_so_far:
            keep_sorted_positions.append(pos)
            best_rmse_so_far = rmse

    keep_sorted_positions = np.asarray(keep_sorted_positions, dtype=int)
    return sorted_points[keep_sorted_positions], order[keep_sorted_positions]


def _normalize_columns(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(float, copy=True)
    mins = arr.min(axis=0)
    spans = arr.max(axis=0) - mins
    spans[spans == 0.0] = 1.0
    return (arr - mins) / spans


def _bend_angles(front_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(front_xy)
    if n < 3:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)

    interior = np.arange(1, n - 1, dtype=int)
    angles = np.empty(n - 2, dtype=float)

    for k, i in enumerate(interior):
        xL, yL = front_xy[i - 1]
        x, y = front_xy[i]
        xR, yR = front_xy[i + 1]

        theta_L = atan2(yL - y, x - xL)
        theta_R = atan2(y - yR, xR - x)

        angles[k] = theta_L - theta_R

    return interior, angles


def _kmeans_1d(
    values: np.ndarray, k: int, max_iter: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    unique_values = np.unique(values)

    if len(values) == 0:
        raise ValueError("K-means received no values.")
    if k <= 0:
        raise ValueError("k must be positive.")

    k = min(k, len(unique_values))
    if k == 1:
        return np.zeros(len(values), dtype=int), np.array([values.mean()], dtype=float)

    centers = np.quantile(values, np.linspace(0.0, 1.0, k))

    for _ in range(max_iter):
        distances = np.abs(values[:, None] - centers[None, :])
        labels = np.argmin(distances, axis=1)

        new_centers = centers.copy()
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                new_centers[j] = values[mask].mean()

        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    return labels, centers


def select_mckp(
    points: Sequence[tuple[float, float]],
    *,
    k: int = 3,
    assume_pareto_front: bool = False,
    normalize_for_angles: bool = False,
) -> MCKPResult:
    """
    Select a point by Minimal Complexity Knee Point (MCKP).

    Args:
        points:
            Sequence of (rmse, complexity) points. Both are minimized.
        k:
            Number of clusters for K-means on bend angles.
            The paper reports 3 as a reasonable default.
        assume_pareto_front:
            If True, points are assumed already to be the Pareto front.
            If False, the Pareto front is extracted first.
        normalize_for_angles:
            Optional practical improvement. If True, normalize complexity and rmse
            to [0, 1] before angle calculation. The paper visualizes normalized
            objectives in some figures, but Algorithm 1 itself does not require
            this step explicitly.

    Returns:
        MCKPResult
    """

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(
            "points must have shape (n_points, 2) with (rmse, complexity)."
        )
    if len(pts) == 0:
        raise ValueError("points must not be empty.")

    if assume_pareto_front:
        order = np.lexsort((pts[:, 0], pts[:, 1]))
        front = pts[order]
        front_input_idx = order
    else:
        front, front_input_idx = _pareto_front_minimize(pts)

    front_xy = front[:, [1, 0]]  # x=complexity, y=rmse
    if normalize_for_angles:
        front_xy = _normalize_columns(front_xy)

    interior, angles = _bend_angles(front_xy)

    labels = None
    centers = None
    best_cluster = None

    if len(front) < 3:
        chosen_front_idx = 0
        knee_front_indices = np.array([0], dtype=int)

    elif len(angles) < k or len(np.unique(angles)) < k:
        best_pos = int(np.argmax(angles))
        chosen_front_idx = int(interior[best_pos])
        knee_front_indices = np.array([chosen_front_idx], dtype=int)

    else:
        labels, centers = _kmeans_1d(angles, k=k)
        best_cluster = int(np.argmax(centers))

        knee_front_indices = interior[labels == best_cluster]
        if len(knee_front_indices) == 0:
            best_pos = int(np.argmax(angles))
            chosen_front_idx = int(interior[best_pos])
            knee_front_indices = np.array([chosen_front_idx], dtype=int)
        else:
            complexities = front[knee_front_indices, 0]  # --- JK
            chosen_front_idx = int(knee_front_indices[np.argmin(complexities)])

    chosen_point = tuple(front[chosen_front_idx])
    chosen_original_index = int(front_input_idx[chosen_front_idx])

    return MCKPResult(
        point=chosen_point,
        original_index=chosen_original_index,
        front_points=front,
        front_input_indices=front_input_idx,
        knee_front_indices=knee_front_indices,
        interior_front_indices=interior,
        bend_angles=angles,
        angle_cluster_labels=labels,
        angle_cluster_centers=centers,
        selected_cluster=best_cluster,
    )


def get_points(seed_folder: Path, cplx_type: str = "complexity_nb_ops"):
    individuals = []
    for model in seed_folder.iterdir():
        if model.suffix != ".m":
            continue  # --- Consider only Matlab *.m files
        valid_loss, rmse_constr, complexity_nn, expr = parse_individual_file(model)
        # --- Count operators and functions in the raw expression
        simplified_expr = matlab_to_python(expr)
        simplified_expr = simplify_expression_to_python(
            orig_expr=simplified_expr, simplification_method=None, timeout=5
        )
        complexity_nb_ops, _ = count_ops_with_breakdown(simplified_expr)
        # ---
        expr = python_to_matlab(simplified_expr)
        if valid_loss is not None and rmse_constr is not None:
            individuals.append(
                {
                    "name": model.name,
                    "valid_loss": valid_loss,
                    "rmse_constr": rmse_constr,
                    "complexity_nn": complexity_nn,
                    "complexity_nb_ops": complexity_nb_ops,
                    "expr": expr,
                }
            )
    rmse_valid_values = [ind["valid_loss"] for ind in individuals]
    complexity_values = [ind[cplx_type] for ind in individuals]
    points = [(r, c) for r, c in zip(rmse_valid_values, complexity_values)]
    return points, individuals
