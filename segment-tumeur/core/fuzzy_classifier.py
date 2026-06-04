"""Classifieur flou supervisé (sans deep learning) basé sur prototypes.

Approche:
- On apprend, pour chaque classe, un prototype moyen (mu) et une dispersion (sigma)
  sur des features interprétables.
- En prédiction, on calcule une compatibilité floue de type gaussienne:
    score_c = exp(-0.5 * Σ_k ((x_k - mu_ck)/sigma_ck)^2)
- Les scores sont normalisés en probabilités.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


CLASSES = ["no tumor", "glioma", "meningioma", "pituitary tumor"]


@dataclass
class FuzzyClassifierModel:
    """Paramètres appris du classifieur flou."""

    classes: list[str]
    mu: np.ndarray  # shape: (n_classes, n_features)
    sigma: np.ndarray  # shape: (n_classes, n_features)
    feature_names: list[str]

    def predict_proba(self, feature_vector: np.ndarray) -> dict[str, float]:
        """Calcule P(classe | x) à partir des scores flous."""
        x = feature_vector.astype(np.float64).reshape(1, -1)
        z = (x - self.mu) / self.sigma
        dist2 = np.sum(z * z, axis=1)
        raw_scores = np.exp(-0.5 * dist2)
        raw_scores = np.clip(raw_scores, 1e-12, None)
        probs = raw_scores / np.sum(raw_scores)
        return {label: float(prob) for label, prob in zip(self.classes, probs)}

    def predict(self, feature_vector: np.ndarray) -> tuple[str, float]:
        """Retourne (classe, confiance)."""
        proba = self.predict_proba(feature_vector)
        label = max(proba, key=proba.get)
        return label, proba[label]

    def to_json_dict(self) -> dict:
        """Sérialise le modèle en dictionnaire JSON."""
        return {
            "classes": self.classes,
            "feature_names": self.feature_names,
            "mu": self.mu.tolist(),
            "sigma": self.sigma.tolist(),
        }

    @classmethod
    def from_json_dict(cls, payload: dict) -> "FuzzyClassifierModel":
        """Désérialise un modèle depuis JSON."""
        return cls(
            classes=list(payload["classes"]),
            feature_names=list(payload["feature_names"]),
            mu=np.asarray(payload["mu"], dtype=np.float64),
            sigma=np.asarray(payload["sigma"], dtype=np.float64),
        )

    def save(self, path: str | Path) -> None:
        """Sauvegarde le modèle sur disque."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_json_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FuzzyClassifierModel":
        """Charge un modèle depuis disque."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_json_dict(payload)


def fit_fuzzy_classifier(features: np.ndarray, labels: list[str], feature_names: list[str]) -> FuzzyClassifierModel:
    """Apprend mu/sigma par classe à partir d'un jeu annoté."""
    if features.ndim != 2:
        raise ValueError("features doit être 2D (n_samples, n_features).")
    if len(labels) != features.shape[0]:
        raise ValueError("labels et features n'ont pas la même taille.")

    classes = sorted(set(labels))
    if len(classes) < 2:
        raise ValueError("Le dataset doit contenir au moins 2 classes.")

    mu_list: list[np.ndarray] = []
    sigma_list: list[np.ndarray] = []
    for class_name in classes:
        idx = [i for i, lbl in enumerate(labels) if lbl == class_name]
        subset = features[idx]
        mu = np.mean(subset, axis=0)
        sigma = np.std(subset, axis=0)
        sigma = np.clip(sigma, 1e-3, None)
        mu_list.append(mu)
        sigma_list.append(sigma)

    return FuzzyClassifierModel(
        classes=classes,
        mu=np.vstack(mu_list),
        sigma=np.vstack(sigma_list),
        feature_names=feature_names,
    )
