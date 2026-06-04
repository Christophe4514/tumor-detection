"""Utilitaires de métriques pour l'évaluation de la segmentation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunMetrics:
    """Métriques globales d'un lancement FCM."""

    iterations: int
    execution_time_sec: float
    final_objective_value: float
    converged: bool


def build_metrics(iterations: int, execution_time_sec: float, objective_history: list[float], converged: bool) -> RunMetrics:
    """Construit un objet métrique à partir des sorties FCM."""
    final_value = objective_history[-1] if objective_history else float("nan")
    return RunMetrics(
        iterations=iterations,
        execution_time_sec=execution_time_sec,
        final_objective_value=final_value,
        converged=converged,
    )
