"""Prétraitement des IRM pour la segmentation floue.

Contraintes respectées:
- OpenCV pour le traitement d'image.
- NumPy pour les opérations matricielles.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PreprocessingResult:
    """Conteneur des sorties de prétraitement."""

    gray_image: np.ndarray
    median_filtered: np.ndarray
    normalized_image: np.ndarray


def to_grayscale(image_bgr: np.ndarray) -> np.ndarray:
    """Convertit une image BGR en niveaux de gris.

    Args:
        image_bgr: Image couleur lue avec OpenCV (format BGR).

    Returns:
        Image en niveaux de gris, type uint8.
    """
    if image_bgr is None:
        raise ValueError("L'image d'entrée est vide.")
    if image_bgr.ndim == 2:
        return image_bgr.copy()
    if image_bgr.ndim != 3:
        raise ValueError("Format d'image non supporté pour la conversion en gris.")

    n_channels = image_bgr.shape[2]
    if n_channels == 3:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if n_channels == 4:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2GRAY)

    raise ValueError("Nombre de canaux non supporté. Attendu: 1, 3 ou 4 canaux.")


def apply_median_filter(gray_image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Applique un filtre médian pour réduire le bruit impulsionnel.

    Le filtre médian est fréquent en imagerie médicale car il préserve
    mieux les contours qu'un filtre linéaire simple.
    """
    if kernel_size % 2 == 0:
        raise ValueError("Le kernel du filtre médian doit être impair.")
    if gray_image.ndim != 2:
        raise ValueError("Le filtre médian attend une image en niveaux de gris (2D).")

    height, width = gray_image.shape[:2]
    min_dim = min(height, width)

    # OpenCV peut échouer si le kernel est plus grand que l'image.
    # On ajuste dynamiquement à la plus grande valeur impaire valide.
    if min_dim < 3:
        return gray_image.copy()

    adjusted_kernel = min(kernel_size, min_dim)
    if adjusted_kernel % 2 == 0:
        adjusted_kernel -= 1
    adjusted_kernel = max(3, adjusted_kernel)

    return cv2.medianBlur(gray_image, adjusted_kernel)


def normalize_to_unit_interval(image_uint8: np.ndarray) -> np.ndarray:
    """Normalise l'image dans [0, 1] (float64).

    Cette normalisation homogénéise l'échelle des intensités pour
    stabiliser les calculs de l'algorithme FCM.
    """
    image_float = image_uint8.astype(np.float64)
    return image_float / 255.0


def preprocess_mri(image_bgr: np.ndarray, median_kernel_size: int = 5) -> PreprocessingResult:
    """Pipeline complet de prétraitement IRM.

    Étapes:
    1) Conversion en niveaux de gris
    2) Filtrage médian
    3) Normalisation dans [0, 1]
    """
    gray = to_grayscale(image_bgr)
    denoised = apply_median_filter(gray, kernel_size=median_kernel_size)
    normalized = normalize_to_unit_interval(denoised)
    return PreprocessingResult(gray_image=gray, median_filtered=denoised, normalized_image=normalized)
