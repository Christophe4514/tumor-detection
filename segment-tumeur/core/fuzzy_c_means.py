"""Implémentation manuelle de Fuzzy C-Means (FCM) avec NumPy uniquement.

Ce module suit les équations classiques de la logique floue:

Fonction objectif:
    J_m(U, V) = Σ_i Σ_j (u_ij^m) * ||x_i - v_j||^2

Mise à jour des centres:
    v_j = [Σ_i (u_ij^m * x_i)] / [Σ_i u_ij^m]

Mise à jour des appartenances:
    u_ij = 1 / Σ_k ( ||x_i - v_j|| / ||x_i - v_k|| )^(2/(m-1))

Où:
- x_i: i-ème pixel (ou vecteur de caractéristiques),
- v_j: centre du cluster j,
- u_ij: degré d'appartenance de x_i au cluster j,
- m > 1: coefficient de flou (souvent m = 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional
import time

import numpy as np


@dataclass
class FCMResult:
    """Sorties complètes de l'algorithme FCM."""

    membership: np.ndarray
    centers: np.ndarray
    objective_history: list[float]
    n_iterations: int
    elapsed_time_sec: float
    converged: bool


class FuzzyCMeans:
    """Algorithme Fuzzy C-Means codé manuellement."""

    def __init__(
        self,
        n_clusters: int = 4,
        m: float = 2.0,
        epsilon: float = 1e-3,
        max_iterations: int = 100,
        random_state: Optional[int] = 42,
    ) -> None:
        if n_clusters < 2:
            raise ValueError("Le nombre de clusters doit être >= 2.")
        if m <= 1.0:
            raise ValueError("Le paramètre m doit être > 1.")
        if epsilon <= 0:
            raise ValueError("epsilon doit être strictement positif.")
        if max_iterations < 1:
            raise ValueError("max_iterations doit être >= 1.")

        self.n_clusters = n_clusters
        self.m = m
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.random_state = random_state

    def _initialize_membership(self, n_samples: int) -> np.ndarray:
        """Initialise U aléatoirement puis normalise chaque ligne à 1."""
        rng = np.random.default_rng(self.random_state)
        membership = rng.random((n_samples, self.n_clusters))
        membership /= np.sum(membership, axis=1, keepdims=True)
        return membership

    @staticmethod
    def _euclidean_distances_squared(data: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Calcule ||x_i - v_j||^2 pour tous (i, j)."""
        # Broadcasting:
        # data:    (n_samples, n_features)
        # centers: (n_clusters, n_features)
        # diff:    (n_samples, n_clusters, n_features)
        diff = data[:, np.newaxis, :] - centers[np.newaxis, :, :]
        return np.sum(diff * diff, axis=2)

    def _update_centers(self, data: np.ndarray, membership: np.ndarray) -> np.ndarray:
        """Met à jour v_j selon la formule FCM."""
        um = membership ** self.m  # u_ij^m
        numerator = um.T @ data  # Σ_i (u_ij^m * x_i), forme (n_clusters, n_features)
        denominator = np.sum(um, axis=0)[:, np.newaxis]  # Σ_i u_ij^m, forme (n_clusters, 1)
        denominator = np.clip(denominator, 1e-12, None)  # stabilité numérique
        return numerator / denominator

    def _update_membership(self, distances_sq: np.ndarray) -> np.ndarray:
        """Met à jour u_ij selon la formule FCM.

        distances_sq[i, j] = ||x_i - v_j||^2
        On utilise distances non-carrées dans la formule ratio, mais
        (d_j / d_k)^(2/(m-1)) = (d_j^2 / d_k^2)^(1/(m-1)),
        donc on peut travailler directement avec distances_sq.
        """
        power = 1.0 / (self.m - 1.0)
        eps = 1e-12

        # Implémentation vectorisée:
        # u_ij = d_ij^(-1/(m-1)) / Σ_k d_ik^(-1/(m-1))
        # (en travaillant avec d^2, l'exposant devient 1/(m-1)).
        safe_distances = np.clip(distances_sq, eps, None)
        inv_dist_power = safe_distances ** (-power)
        denominator = np.sum(inv_dist_power, axis=1, keepdims=True)
        membership_new = inv_dist_power / denominator

        # Cas particulier robuste: x_i exactement sur un ou plusieurs centres.
        # On répartit uniformément la masse sur les centres à distance nulle.
        zero_mask = distances_sq < eps
        rows_with_zero = np.any(zero_mask, axis=1)
        if np.any(rows_with_zero):
            membership_new[rows_with_zero] = 0.0
            zero_counts = np.sum(zero_mask[rows_with_zero], axis=1, keepdims=True)
            membership_new[rows_with_zero] = zero_mask[rows_with_zero] / zero_counts

        return membership_new.astype(np.float64, copy=False)

    def _objective_function(self, membership: np.ndarray, distances_sq: np.ndarray) -> float:
        """Calcule la fonction objectif J_m(U, V)."""
        return float(np.sum((membership ** self.m) * distances_sq))

    def fit(
        self,
        data: np.ndarray,
        callback: Optional[Callable[[int, float], None]] = None,
    ) -> FCMResult:
        """Exécute l'algorithme FCM.

        Args:
            data: Matrice (n_samples, n_features). Pour une image grayscale,
                on peut fournir les intensités aplaties en (N, 1).
            callback: Fonction optionnelle appelée à chaque itération avec
                (itération, valeur de J_m). Utile pour l'affichage progressif.
        """
        if data.ndim != 2:
            raise ValueError("data doit être de forme (n_samples, n_features).")

        n_samples = data.shape[0]
        membership = self._initialize_membership(n_samples)
        objective_history: list[float] = []
        converged = False
        start_time = time.perf_counter()

        for iteration in range(1, self.max_iterations + 1):
            centers = self._update_centers(data, membership)
            distances_sq = self._euclidean_distances_squared(data, centers)
            membership_new = self._update_membership(distances_sq)
            objective_value = self._objective_function(membership_new, distances_sq)
            objective_history.append(objective_value)

            if callback is not None:
                callback(iteration, objective_value)

            # Critère de convergence:
            # ||U^(t+1) - U^(t)|| < epsilon
            delta_u = np.linalg.norm(membership_new - membership)
            membership = membership_new

            if delta_u < self.epsilon:
                converged = True
                break

        elapsed = time.perf_counter() - start_time
        return FCMResult(
            membership=membership,
            centers=centers,
            objective_history=objective_history,
            n_iterations=len(objective_history),
            elapsed_time_sec=elapsed,
            converged=converged,
        )
