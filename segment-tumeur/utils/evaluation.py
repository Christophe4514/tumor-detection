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
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    per_class_f1: dict[str, float]
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
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
    per_class_precision: dict[str, float] = {}
    per_class_recall: dict[str, float] = {}
    per_class_f1: dict[str, float] = {}
    f1_weighted_sum = 0.0
    for i, label in enumerate(labels):
        row_sum = int(np.sum(cm[i, :]))
        col_sum = int(np.sum(cm[:, i]))
        tp = int(cm[i, i])
        precision = float(tp / col_sum) if col_sum > 0 else 0.0
        recall = float(tp / row_sum) if row_sum > 0 else 0.0
        f1 = float((2.0 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_class_accuracy[label] = float(cm[i, i] / row_sum) if row_sum > 0 else 0.0
        per_class_precision[label] = precision
        per_class_recall[label] = recall
        per_class_f1[label] = f1
        f1_weighted_sum += f1 * row_sum

    macro_precision = float(np.mean(list(per_class_precision.values()))) if labels else 0.0
    macro_recall = float(np.mean(list(per_class_recall.values()))) if labels else 0.0
    macro_f1 = float(np.mean(list(per_class_f1.values()))) if labels else 0.0
    weighted_f1 = float(f1_weighted_sum / total) if total > 0 else 0.0

    return EvaluationReport(
        labels=labels,
        confusion_matrix=cm,
        accuracy=accuracy,
        per_class_accuracy=per_class_accuracy,
        per_class_precision=per_class_precision,
        per_class_recall=per_class_recall,
        per_class_f1=per_class_f1,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
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
    lines.append(f"Macro Precision: {report.macro_precision:.4f}")
    lines.append(f"Macro Recall: {report.macro_recall:.4f}")
    lines.append(f"Macro F1-score: {report.macro_f1:.4f}")
    lines.append(f"Weighted F1-score: {report.weighted_f1:.4f}")
    lines.append("")
    for label in labels:
        lines.append(f"Accuracy {label}: {report.per_class_accuracy[label]:.4f}")
        lines.append(
            f"  Precision={report.per_class_precision[label]:.4f} "
            f"Recall={report.per_class_recall[label]:.4f} "
            f"F1={report.per_class_f1[label]:.4f}"
        )
    return "\n".join(lines)
