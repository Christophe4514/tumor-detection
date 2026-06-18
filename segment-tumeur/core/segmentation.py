"""Segmentation d'image IRM via Fuzzy C-Means."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from core.fuzzy_c_means import FCMResult, FuzzyCMeans


@dataclass
class SegmentationResult:
    """Sorties de la segmentation floue."""

    fcm_result: FCMResult
    labels: np.ndarray
    segmented_image_uint8: np.ndarray
    cluster_masks: list[np.ndarray]
    cluster_centers: np.ndarray


def _labels_to_segmented_image(labels: np.ndarray, centers: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Crée une image segmentée en remplaçant chaque pixel par son centre de cluster."""
    centers_1d = centers.reshape(-1)
    segmented_flat = centers_1d[labels]
    segmented = segmented_flat.reshape(shape)
    return np.clip(segmented * 255.0, 0, 255).astype(np.uint8)


def segment_with_fcm(
    normalized_image: np.ndarray,
    n_clusters: int = 4,
    m: float = 2.0,
    epsilon: float = 1e-3,
    max_iterations: int = 100,
    random_state: int = 42,
    use_mpi: Optional[bool] = None,
    callback: Optional[Callable[[int, float], None]] = None,
) -> SegmentationResult:
    """Segmente une image normalisée [0, 1] par FCM.

    Étapes:
    1) Aplatissement de l'image en vecteur de pixels (N, 1).
    2) Exécution de FCM.
    3) Affectation dure via argmax(u_ij).
    4) Reconstruction de l'image segmentée et des masques de clusters.
    """
    if normalized_image.ndim != 2:
        raise ValueError("L'image normalisée doit être en 2D (grayscale).")

    original_shape = normalized_image.shape
    data = normalized_image.reshape(-1, 1)

    fcm = FuzzyCMeans(
        n_clusters=n_clusters,
        m=m,
        epsilon=epsilon,
        max_iterations=max_iterations,
        random_state=random_state,
        use_mpi=use_mpi,
    )
    fcm_result = fcm.fit(data, callback=callback)

    # Affectation dure: cluster dont le degré d'appartenance est maximal.
    labels = np.argmax(fcm_result.membership, axis=1)

    segmented_image = _labels_to_segmented_image(labels, fcm_result.centers, original_shape)

    cluster_masks: list[np.ndarray] = []
    labels_2d = labels.reshape(original_shape)
    for cluster_idx in range(n_clusters):
        mask = (labels_2d == cluster_idx).astype(np.uint8) * 255
        cluster_masks.append(mask)

    return SegmentationResult(
        fcm_result=fcm_result,
        labels=labels_2d,
        segmented_image_uint8=segmented_image,
        cluster_masks=cluster_masks,
        cluster_centers=fcm_result.centers.reshape(-1),
    )
