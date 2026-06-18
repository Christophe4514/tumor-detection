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
    feature_mean: np.ndarray | None = None
    feature_std: np.ndarray | None = None
    class_priors: np.ndarray | None = None
    precision_matrices: np.ndarray | None = None  # shape: (n_classes, n_features, n_features)
    log_det_cov: np.ndarray | None = None  # shape: (n_classes,)

    def predict_proba(self, feature_vector: np.ndarray) -> dict[str, float]:
        """Calcule P(classe | x) à partir des scores flous."""
        x = feature_vector.astype(np.float64).reshape(1, -1)
        if self.feature_mean is not None and self.feature_std is not None:
            x = (x - self.feature_mean.reshape(1, -1)) / self.feature_std.reshape(1, -1)
        if (
            self.precision_matrices is not None
            and self.log_det_cov is not None
            and self.precision_matrices.shape[0] == self.mu.shape[0]
        ):
            n_features = x.shape[1]
            log_scores: list[float] = []
            for class_idx in range(self.mu.shape[0]):
                delta = (x - self.mu[class_idx].reshape(1, -1)).reshape(-1)
                precision = self.precision_matrices[class_idx]
                quad = float(delta.T @ precision @ delta)
                logp = -0.5 * (quad + float(self.log_det_cov[class_idx]) + n_features * np.log(2.0 * np.pi))
                if self.class_priors is not None and self.class_priors.shape[0] == self.mu.shape[0]:
                    logp += float(np.log(max(self.class_priors[class_idx], 1e-12)))
                log_scores.append(logp)
            log_scores_arr = np.asarray(log_scores, dtype=np.float64)
            log_scores_arr -= np.max(log_scores_arr)
            raw_scores = np.exp(log_scores_arr)
        else:
            z = (x - self.mu) / self.sigma
            dist2 = np.sum(z * z, axis=1)
            raw_scores = np.exp(-0.5 * dist2)
            if self.class_priors is not None and self.class_priors.shape[0] == raw_scores.shape[0]:
                raw_scores = raw_scores * self.class_priors

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
            "feature_mean": self.feature_mean.tolist() if self.feature_mean is not None else None,
            "feature_std": self.feature_std.tolist() if self.feature_std is not None else None,
            "class_priors": self.class_priors.tolist() if self.class_priors is not None else None,
            "precision_matrices": self.precision_matrices.tolist() if self.precision_matrices is not None else None,
            "log_det_cov": self.log_det_cov.tolist() if self.log_det_cov is not None else None,
        }

    @classmethod
    def from_json_dict(cls, payload: dict) -> "FuzzyClassifierModel":
        """Désérialise un modèle depuis JSON."""
        return cls(
            classes=list(payload["classes"]),
            feature_names=list(payload["feature_names"]),
            mu=np.asarray(payload["mu"], dtype=np.float64),
            sigma=np.asarray(payload["sigma"], dtype=np.float64),
            feature_mean=(
                np.asarray(payload["feature_mean"], dtype=np.float64)
                if payload.get("feature_mean") is not None
                else None
            ),
            feature_std=(
                np.asarray(payload["feature_std"], dtype=np.float64)
                if payload.get("feature_std") is not None
                else None
            ),
            class_priors=(
                np.asarray(payload["class_priors"], dtype=np.float64)
                if payload.get("class_priors") is not None
                else None
            ),
            precision_matrices=(
                np.asarray(payload["precision_matrices"], dtype=np.float64)
                if payload.get("precision_matrices") is not None
                else None
            ),
            log_det_cov=(
                np.asarray(payload["log_det_cov"], dtype=np.float64)
                if payload.get("log_det_cov") is not None
                else None
            ),
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

    feature_mean = np.mean(features, axis=0)
    feature_std = np.std(features, axis=0)
    feature_std = np.clip(feature_std, 1e-6, None)
    features_norm = (features - feature_mean) / feature_std

    mu_list: list[np.ndarray] = []
    sigma_list: list[np.ndarray] = []
    prior_list: list[float] = []
    precision_list: list[np.ndarray] = []
    log_det_cov_list: list[float] = []
    n_features = features_norm.shape[1]
    reg_lambda = 0.10
    for class_name in classes:
        idx = [i for i, lbl in enumerate(labels) if lbl == class_name]
        subset = features_norm[idx]
        mu = np.mean(subset, axis=0)
        sigma = np.std(subset, axis=0)
        sigma = np.clip(sigma, 0.05, None)
        if subset.shape[0] > 1:
            cov = np.cov(subset, rowvar=False)
            if cov.ndim == 0:
                cov = np.asarray([[float(cov)]], dtype=np.float64)
        else:
            cov = np.eye(n_features, dtype=np.float64)
        cov = np.asarray(cov, dtype=np.float64)
        cov_reg = cov + (reg_lambda * np.eye(n_features, dtype=np.float64))
        precision = np.linalg.pinv(cov_reg)
        sign, logdet = np.linalg.slogdet(cov_reg)
        if sign <= 0:
            logdet = float(np.log(np.maximum(np.linalg.det(cov_reg), 1e-12)))
        mu_list.append(mu)
        sigma_list.append(sigma)
        prior_list.append(float(len(idx) / len(labels)))
        precision_list.append(precision)
        log_det_cov_list.append(float(logdet))

    return FuzzyClassifierModel(
        classes=classes,
        mu=np.vstack(mu_list),
        sigma=np.vstack(sigma_list),
        feature_names=feature_names,
        feature_mean=feature_mean.astype(np.float64),
        feature_std=feature_std.astype(np.float64),
        class_priors=np.asarray(prior_list, dtype=np.float64),
        precision_matrices=np.stack(precision_list, axis=0),
        log_det_cov=np.asarray(log_det_cov_list, dtype=np.float64),
    )
