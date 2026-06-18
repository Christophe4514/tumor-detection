"""Extraction de caractéristiques pour la classification tumorale.

Ce module calcule un petit vecteur de features interprétables à partir:
- du masque tumoral raffiné,
- de l'image prétraitée (grayscale uint8).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class TumorFeatures:
    """Features morphologiques/intensité de la lésion."""

    area_ratio: float
    edge_distance: float
    circularity: float
    irregularity: float
    intensity_norm: float
    bbox_fill_ratio: float
    intensity_std_norm: float
    intensity_p90_norm: float
    centroid_x_norm: float
    centroid_y_norm: float

    def as_array(self) -> np.ndarray:
        """Convertit vers un vecteur NumPy (5 dimensions)."""
        return np.asarray(
            [
                self.area_ratio,
                self.edge_distance,
                self.circularity,
                self.irregularity,
                self.intensity_norm,
                self.bbox_fill_ratio,
                self.intensity_std_norm,
                self.intensity_p90_norm,
                self.centroid_x_norm,
                self.centroid_y_norm,
            ],
            dtype=np.float64,
        )


def extract_tumor_features(refined_mask: np.ndarray, preprocessed_gray_uint8: np.ndarray) -> TumorFeatures:
    """Extrait les features de la plus grande composante connexe."""
    binary = (refined_mask > 0).astype(np.uint8)
    area_ratio = float(np.mean(binary))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1 or area_ratio < 1e-6:
        return TumorFeatures(
            area_ratio=0.0,
            edge_distance=0.0,
            circularity=0.0,
            irregularity=1.0,
            intensity_norm=0.0,
            bbox_fill_ratio=0.0,
            intensity_std_norm=0.0,
            intensity_p90_norm=0.0,
            centroid_x_norm=0.0,
            centroid_y_norm=0.0,
        )

    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_component_idx = int(np.argmax(component_areas)) + 1
    largest_mask = (labels == largest_component_idx).astype(np.uint8)
    largest_area_ratio = float(np.mean(largest_mask))

    ys, xs = np.where(largest_mask > 0)
    cx_norm = float(np.mean(xs) / max(1, refined_mask.shape[1] - 1))
    cy_norm = float(np.mean(ys) / max(1, refined_mask.shape[0] - 1))
    edge_distance = float(min(cx_norm, 1.0 - cx_norm, cy_norm, 1.0 - cy_norm))
    area_px = float(np.sum(largest_mask))

    x_min, x_max = int(np.min(xs)), int(np.max(xs))
    y_min, y_max = int(np.min(ys)), int(np.max(ys))
    bbox_w = max(1, x_max - x_min + 1)
    bbox_h = max(1, y_max - y_min + 1)
    bbox_area = float(bbox_w * bbox_h)
    bbox_fill_ratio = float(area_px / bbox_area) if bbox_area > 0 else 0.0

    contours, _ = cv2.findContours((largest_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = float(cv2.arcLength(contours[0], True)) if contours else 0.0
    circularity = float(max(0.0, min(1.0, (4.0 * np.pi * area_px) / (perimeter * perimeter + 1e-9))))
    irregularity = float(1.0 - circularity)

    tumor_pixels = preprocessed_gray_uint8[largest_mask > 0]
    mean_intensity = float(np.mean(tumor_pixels)) if tumor_pixels.size > 0 else 0.0
    std_intensity = float(np.std(tumor_pixels)) if tumor_pixels.size > 0 else 0.0
    p90_intensity = float(np.percentile(tumor_pixels, 90.0)) if tumor_pixels.size > 0 else 0.0
    intensity_norm = mean_intensity / 255.0

    return TumorFeatures(
        area_ratio=largest_area_ratio,
        edge_distance=edge_distance,
        circularity=circularity,
        irregularity=irregularity,
        intensity_norm=intensity_norm,
        bbox_fill_ratio=float(np.clip(bbox_fill_ratio, 0.0, 1.0)),
        intensity_std_norm=float(np.clip(std_intensity / 255.0, 0.0, 1.0)),
        intensity_p90_norm=float(np.clip(p90_intensity / 255.0, 0.0, 1.0)),
        centroid_x_norm=float(np.clip(cx_norm, 0.0, 1.0)),
        centroid_y_norm=float(np.clip(cy_norm, 0.0, 1.0)),
    )
