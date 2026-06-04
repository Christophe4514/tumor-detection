"""Évaluation de classification: accuracy et matrice de confusion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EvaluationReport:
    """Rapport d'évaluation global."""

    labels: list[str]
    confusion_matrix: np.ndarray
    accuracy: float
    per_class_accuracy: dict[str, float]
    n_test_samples: int


def compute_evaluation_report(y_true: list[str], y_pred: list[str], labels: list[str]) -> EvaluationReport:
    """Construit la matrice de confusion et les métriques."""
    n = len(labels)
    index = {label: i for i, label in enumerate(labels)}
    cm = np.zeros((n, n), dtype=np.int64)

    for yt, yp in zip(y_true, y_pred):
        if yt in index and yp in index:
            cm[index[yt], index[yp]] += 1

    total = int(np.sum(cm))
    correct = int(np.trace(cm))
    accuracy = float(correct / total) if total > 0 else 0.0

    per_class_accuracy: dict[str, float] = {}
    for i, label in enumerate(labels):
        row_sum = int(np.sum(cm[i, :]))
        per_class_accuracy[label] = float(cm[i, i] / row_sum) if row_sum > 0 else 0.0

    return EvaluationReport(
        labels=labels,
        confusion_matrix=cm,
        accuracy=accuracy,
        per_class_accuracy=per_class_accuracy,
        n_test_samples=total,
    )


def format_confusion_matrix_text(report: EvaluationReport) -> str:
    """Formate une matrice de confusion lisible en texte."""
    labels = report.labels
    cm = report.confusion_matrix
    col_width = max(12, max(len(label) for label in labels) + 2)

    header = " " * col_width + "".join(label.ljust(col_width) for label in labels)
    lines = [header, "-" * len(header)]

    for i, row_label in enumerate(labels):
        row_vals = "".join(str(int(cm[i, j])).ljust(col_width) for j in range(len(labels)))
        lines.append(row_label.ljust(col_width) + row_vals)

    lines.append("")
    lines.append(f"Accuracy globale: {report.accuracy:.4f} ({report.n_test_samples} échantillons test)")
    for label in labels:
        lines.append(f"Accuracy {label}: {report.per_class_accuracy[label]:.4f}")
    return "\n".join(lines)
