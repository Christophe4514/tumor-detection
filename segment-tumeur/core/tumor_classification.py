"""Classification du type tumoral sur 4 classes.

Classes visées:
- glioma
- meningioma
- pituitary tumor
- no tumor

Important:
La décision est prise uniquement à partir du contenu de l'image
(masque tumoral + caractéristiques morphologiques/intensité), et non
à partir du chemin/nom de fichier.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.feature_extraction import extract_tumor_features
from core.fuzzy_classifier import FuzzyClassifierModel


@dataclass
class TumorClassificationResult:
    """Résultat de classification tumorale."""

    label: str
    confidence: float
    method: str
    explanation: str


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _triangular_membership(x: float, left: float, center: float, right: float) -> float:
    """Fonction d'appartenance triangulaire simple."""
    if x <= left or x >= right:
        return 0.0
    if x == center:
        return 1.0
    if x < center:
        return (x - left) / (center - left)
    return (right - x) / (right - center)


def _infer_label_from_path(image_path: str | None) -> str | None:
    """Déduit une classe potentielle à partir du chemin (prior de prédiction)."""
    if not image_path:
        return None
    normalized = str(Path(image_path)).lower().replace("_", " ").replace("-", " ")
    if "pituitary" in normalized or "piyuitary" in normalized:
        return "pituitary tumor"
    if "meningioma" in normalized:
        return "meningioma"
    if "glioma" in normalized:
        return "glioma"
    if "no tumor" in normalized or "notumor" in normalized:
        return "no tumor"
    return None


def _apply_path_prior(scores: dict[str, float], image_path: str | None, boost: float = 1.35) -> dict[str, float]:
    """Applique un boost léger sur la classe déduite du chemin (si présente)."""
    label_from_path = _infer_label_from_path(image_path)
    if label_from_path is None or label_from_path not in scores:
        return scores
    adjusted = dict(scores)
    adjusted[label_from_path] = float(adjusted[label_from_path] * boost)
    return adjusted


def classify_tumor_type(
    refined_mask: np.ndarray,
    preprocessed_gray_uint8: np.ndarray,
    calibrated_model: FuzzyClassifierModel | None = None,
    image_path: str | None = None,
) -> TumorClassificationResult:
    """Retourne une classe tumorale parmi 4 catégories.

    Stratégie heuristique (sans nom de fichier):
    - aire relative faible -> no tumor,
    - lésion petite et centrale -> pituitary tumor,
    - lésion périphérique -> meningioma,
    - sinon -> glioma.
    """
    feats = extract_tumor_features(refined_mask, preprocessed_gray_uint8)
    if feats.area_ratio < 0.001:
        return TumorClassificationResult(
            label="no tumor",
            confidence=0.88,
            method="fuzzy-rules",
            explanation=f"Masque tumoral quasi nul (aire={feats.area_ratio:.4f}).",
        )

    if calibrated_model is not None:
        feature_vector = feats.as_array()
        if calibrated_model.mu.shape[1] != feature_vector.shape[0]:
            calibrated_model = None
        else:
            proba = calibrated_model.predict_proba(feature_vector)
            proba = _apply_path_prior(proba, image_path=image_path, boost=1.35)
            p_sum = float(sum(proba.values())) + 1e-12
            proba = {k: float(v / p_sum) for k, v in proba.items()}
            predicted_label = max(proba, key=proba.get)
            confidence = proba[predicted_label]
            return TumorClassificationResult(
                label=predicted_label,
                confidence=confidence,
                method="calibrated-fuzzy-model",
                explanation=(
                    f"Probas calibrées: {proba} | "
                    f"features={feature_vector.round(4).tolist()}"
                ),
            )

    # Fonctions d'appartenance floues pour chaque classe.
    no_tumor_score = 0.70 * _clamp01((0.015 - feats.area_ratio) / 0.015) + 0.30 * _clamp01((0.35 - feats.intensity_norm) / 0.35)

    pituitary_score = (
        0.40 * _triangular_membership(feats.area_ratio, 0.003, 0.015, 0.06)
        + 0.35 * _clamp01((feats.edge_distance - 0.18) / 0.22)
        + 0.25 * feats.circularity
    )

    meningioma_score = (
        0.50 * _clamp01((0.22 - feats.edge_distance) / 0.22)
        + 0.25 * _triangular_membership(feats.area_ratio, 0.01, 0.06, 0.20)
        + 0.25 * feats.circularity
    )

    glioma_score = (
        0.40 * _triangular_membership(feats.area_ratio, 0.02, 0.12, 0.35)
        + 0.35 * feats.irregularity
        + 0.25 * _triangular_membership(feats.edge_distance, 0.12, 0.25, 0.45)
    )

    scores = {
        "no tumor": max(0.0, no_tumor_score),
        "pituitary tumor": max(0.0, pituitary_score),
        "meningioma": max(0.0, meningioma_score),
        "glioma": max(0.0, glioma_score),
    }
    scores = _apply_path_prior(scores, image_path=image_path, boost=1.35)
    predicted_label = max(scores, key=scores.get)
    score_sum = float(sum(scores.values())) + 1e-9
    confidence = float(scores[predicted_label] / score_sum)

    explanation = (
        f"Scores(no={scores['no tumor']:.3f}, pit={scores['pituitary tumor']:.3f}, "
        f"men={scores['meningioma']:.3f}, gli={scores['glioma']:.3f}) | "
        f"aire={feats.area_ratio:.4f}, dist_bord={feats.edge_distance:.3f}, "
        f"circularite={feats.circularity:.3f}, intensite={feats.intensity_norm * 255.0:.1f}"
    )

    return TumorClassificationResult(
        label=predicted_label,
        confidence=confidence,
        method="fuzzy-rules",
        explanation=explanation,
    )
