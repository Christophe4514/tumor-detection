"""Détection tumorale à partir du résultat de segmentation FCM."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.segmentation import SegmentationResult


@dataclass
class TumorDetectionResult:
    """Sorties de la détection de tumeur."""

    tumor_cluster_index: int
    initial_mask: np.ndarray
    refined_mask: np.ndarray
    extracted_tumor: np.ndarray
    cluster_scores: list[float]


def _morphological_postprocess(mask: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """Applique érosion, dilatation, ouverture, fermeture.

    Ordre pédagogique explicite demandé dans l'énoncé:
    1) Érosion
    2) Dilatation
    3) Ouverture
    4) Fermeture
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=iterations)
    dilated = cv2.dilate(eroded, kernel, iterations=iterations)
    opened = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, kernel, iterations=iterations)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return closed


def _border_contact_ratio(mask: np.ndarray, border_px: int = 5) -> float:
    """Calcule la proportion de pixels du masque en contact avec le bord."""
    binary = (mask > 0).astype(np.uint8)
    total = int(np.sum(binary))
    if total == 0:
        return 1.0

    h, w = binary.shape
    border = np.zeros_like(binary, dtype=np.uint8)
    border[:border_px, :] = 1
    border[-border_px:, :] = 1
    border[:, :border_px] = 1
    border[:, -border_px:] = 1

    touching = int(np.sum((binary == 1) & (border == 1)))
    return float(touching / total)


def _cluster_candidate_score(mask: np.ndarray, gray_uint8: np.ndarray) -> float:
    """Score d'un cluster candidat tumoral.

    Idée:
    - Favoriser les zones d'intensité élevée.
    - Pénaliser fortement les clusters collés au bord (souvent crâne/artefact).
    - Pénaliser légèrement les zones trop grandes.
    """
    binary = (mask > 0).astype(np.uint8)
    area_ratio = float(np.mean(binary))
    if area_ratio <= 1e-6:
        return -1e6

    region_pixels = gray_uint8[binary > 0]
    mean_intensity = float(np.mean(region_pixels)) / 255.0 if region_pixels.size > 0 else 0.0
    border_ratio = _border_contact_ratio(mask, border_px=5)

    score = mean_intensity - (1.25 * border_ratio) - (0.12 * area_ratio)
    return float(score)


def detect_tumor_from_segmentation(
    segmentation_result: SegmentationResult,
    preprocessed_gray_uint8: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1,
) -> TumorDetectionResult:
    """Détecte la tumeur supposée.

    Hypothèse simple (cohérente avec le sujet):
    - La tumeur candidate correspond au cluster de plus forte intensité moyenne.
    """
    # Évalue tous les clusters, au lieu de prendre systématiquement le plus lumineux.
    # Cela évite le biais fréquent vers le crâne/contours brillants.
    scores: list[float] = []
    refined_by_cluster: list[np.ndarray] = []
    for mask in segmentation_result.cluster_masks:
        refined = _morphological_postprocess(mask, kernel_size=kernel_size, iterations=iterations)
        refined_by_cluster.append(refined)
        scores.append(_cluster_candidate_score(refined, preprocessed_gray_uint8))

    tumor_cluster_idx = int(np.argmax(np.asarray(scores, dtype=np.float64)))
    initial_mask = segmentation_result.cluster_masks[tumor_cluster_idx]
    refined_mask = refined_by_cluster[tumor_cluster_idx]

    # Extraction: on conserve uniquement les pixels de l'image prétraitée dans le masque tumoral.
    extracted_tumor = cv2.bitwise_and(preprocessed_gray_uint8, preprocessed_gray_uint8, mask=refined_mask)

    return TumorDetectionResult(
        tumor_cluster_index=tumor_cluster_idx,
        initial_mask=initial_mask,
        refined_mask=refined_mask,
        extracted_tumor=extracted_tumor,
        cluster_scores=scores,
    )
