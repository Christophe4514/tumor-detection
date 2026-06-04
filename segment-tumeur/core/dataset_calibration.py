"""Calibration et évaluation du classifieur flou sur un dataset annoté."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from core.feature_extraction import extract_tumor_features
from core.fuzzy_classifier import FuzzyClassifierModel, fit_fuzzy_classifier
from core.preprocessing import preprocess_mri
from core.segmentation import segment_with_fcm
from core.tumor_detection import detect_tumor_from_segmentation
from utils.evaluation import EvaluationReport, compute_evaluation_report
from utils.image_utils import load_image_bgr


ALLOWED_EXT = {".png", ".jpg", ".jpeg"}
VALID_LABELS = {"no tumor", "glioma", "meningioma", "pituitary tumor"}
FEATURE_NAMES = ["area_ratio", "edge_distance", "circularity", "irregularity", "intensity_norm"]


@dataclass
class CalibrationEvaluationResult:
    """Résultat final calibration + évaluation."""

    model: FuzzyClassifierModel
    report: EvaluationReport
    n_total: int
    n_train: int
    n_test: int
    skipped_files: int


def _infer_label_from_path(path: Path) -> str | None:
    normalized = str(path).lower().replace("_", " ").replace("-", " ")
    if "pituitary" in normalized or "piyuitary" in normalized:
        return "pituitary tumor"
    if "meningioma" in normalized:
        return "meningioma"
    if "glioma" in normalized:
        return "glioma"
    if "no tumor" in normalized or "notumor" in normalized:
        return "no tumor"
    return None


def _collect_images(dataset_dir: Path, max_samples: int | None = None) -> list[tuple[Path, str]]:
    samples: list[tuple[Path, str]] = []
    for path in dataset_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in ALLOWED_EXT:
            continue
        label = _infer_label_from_path(path)
        if label in VALID_LABELS:
            samples.append((path, label))
        if max_samples is not None and len(samples) >= max_samples:
            break
    return samples


def run_calibration_and_evaluation(
    dataset_dir: str | Path,
    n_clusters: int = 4,
    m: float = 2.0,
    epsilon: float = 1e-3,
    max_iterations: int = 80,
    test_ratio: float = 0.25,
    random_state: int = 42,
    max_samples: int | None = 300,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> CalibrationEvaluationResult:
    """Exécute la calibration supervisée puis l'évaluation."""
    ds = Path(dataset_dir)
    if not ds.exists():
        raise ValueError(f"Dossier dataset introuvable: {ds}")

    samples = _collect_images(ds, max_samples=max_samples)
    if len(samples) < 20:
        raise ValueError("Dataset insuffisant: au moins 20 images annotées sont nécessaires.")

    rng = np.random.default_rng(random_state)
    indices = np.arange(len(samples))
    rng.shuffle(indices)
    samples = [samples[i] for i in indices]

    features: list[np.ndarray] = []
    labels: list[str] = []
    skipped = 0
    total = len(samples)

    for idx, (img_path, label) in enumerate(samples, start=1):
        if progress_callback is not None:
            progress_callback(idx, total, f"Traitement: {img_path.name}")
        try:
            image_bgr = load_image_bgr(str(img_path))
            pre = preprocess_mri(image_bgr, median_kernel_size=5)
            seg = segment_with_fcm(
                normalized_image=pre.normalized_image,
                n_clusters=n_clusters,
                m=m,
                epsilon=epsilon,
                max_iterations=max_iterations,
                random_state=random_state,
                callback=None,
            )
            det = detect_tumor_from_segmentation(
                segmentation_result=seg,
                preprocessed_gray_uint8=pre.median_filtered,
                kernel_size=3,
                iterations=1,
            )
            feats = extract_tumor_features(det.refined_mask, pre.median_filtered)
            features.append(feats.as_array())
            labels.append(label)
        except Exception:
            skipped += 1

    if len(features) < 20:
        raise ValueError("Trop peu d'images exploitables après prétraitement/segmentation.")

    X = np.vstack(features)
    y = labels
    n_total = X.shape[0]
    n_test = max(1, int(round(n_total * test_ratio)))
    n_train = n_total - n_test
    if n_train < 10:
        raise ValueError("Pas assez de données pour entraîner le modèle.")

    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]

    model = fit_fuzzy_classifier(X_train, y_train, feature_names=FEATURE_NAMES)
    y_pred: list[str] = []
    for x in X_test:
        pred, _ = model.predict(x)
        y_pred.append(pred)

    report = compute_evaluation_report(y_test, y_pred, labels=sorted(set(y_train) | set(y_test)))
    return CalibrationEvaluationResult(
        model=model,
        report=report,
        n_total=n_total,
        n_train=n_train,
        n_test=n_test,
        skipped_files=skipped,
    )
