"""Calibration et évaluation du classifieur flou sur un dataset annoté."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
import time
from collections import defaultdict

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
FEATURE_NAMES = [
    "area_ratio",
    "edge_distance",
    "circularity",
    "irregularity",
    "intensity_norm",
    "bbox_fill_ratio",
    "intensity_std_norm",
    "intensity_p90_norm",
    "centroid_x_norm",
    "centroid_y_norm",
]


@dataclass
class CalibrationEvaluationResult:
    """Résultat final calibration + évaluation."""

    model: FuzzyClassifierModel
    report: EvaluationReport
    n_total: int
    n_train: int
    n_test: int
    skipped_files: int
    calibration_time_sec: float
    mpi_parallel_images: int
    mpi_max_processes: int


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


def _collect_images(dataset_dir: Path) -> list[tuple[Path, str]]:
    samples: list[tuple[Path, str]] = []
    for path in dataset_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in ALLOWED_EXT:
            continue
        label = _infer_label_from_path(path)
        if label in VALID_LABELS:
            samples.append((path, label))
    return samples


def _stratified_sample(
    samples: list[tuple[Path, str]],
    max_samples: int | None,
    random_state: int,
) -> list[tuple[Path, str]]:
    """Sous-échantillonne de manière stratifiée pour garder plusieurs classes."""
    if max_samples is None or max_samples <= 0 or len(samples) <= max_samples:
        return samples

    rng = np.random.default_rng(random_state)
    by_label: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    for item in samples:
        by_label[item[1]].append(item)

    label_keys = sorted(by_label.keys())
    for label in label_keys:
        arr = by_label[label]
        if len(arr) > 1:
            order = rng.permutation(len(arr))
            by_label[label] = [arr[i] for i in order]

    selected: list[tuple[Path, str]] = []
    per_label_idx = {label: 0 for label in label_keys}
    while len(selected) < max_samples:
        added = False
        for label in label_keys:
            idx = per_label_idx[label]
            if idx < len(by_label[label]):
                selected.append(by_label[label][idx])
                per_label_idx[label] += 1
                added = True
                if len(selected) >= max_samples:
                    break
        if not added:
            break
    return selected


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
    start_time = time.perf_counter()
    ds = Path(dataset_dir)
    if not ds.exists():
        raise ValueError(f"Dossier dataset introuvable: {ds}")

    samples = _collect_images(ds)
    samples = _stratified_sample(samples, max_samples=max_samples, random_state=random_state)
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
    mpi_parallel_images = 0
    mpi_max_processes = 1

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
                use_mpi=None,
                callback=None,
            )
            if seg.fcm_result.mpi_enabled:
                mpi_parallel_images += 1
            mpi_max_processes = max(mpi_max_processes, int(seg.fcm_result.mpi_size))
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
    class_counts: dict[str, int] = {}
    for label in sorted(set(y)):
        class_counts[label] = sum(1 for v in y if v == label)
    if len(class_counts) < 2:
        raise ValueError(
            "Le dataset exploitable après segmentation contient moins de 2 classes. "
            f"Distribution trouvée: {class_counts}. "
            "Augmentez 'Calibration - max samples' ou vérifiez la qualité des images."
        )
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
    calibration_time_sec = float(time.perf_counter() - start_time)
    return CalibrationEvaluationResult(
        model=model,
        report=report,
        n_total=n_total,
        n_train=n_train,
        n_test=n_test,
        skipped_files=skipped,
        calibration_time_sec=calibration_time_sec,
        mpi_parallel_images=mpi_parallel_images,
        mpi_max_processes=mpi_max_processes,
    )
