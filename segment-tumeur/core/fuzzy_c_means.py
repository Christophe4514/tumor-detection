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
import os
import time

import numpy as np

try:
    from mpi4py import MPI  # pyright: ignore[reportMissingImports]
except Exception:  # pragma: no cover - dépend de l'environnement MPI local.
    MPI = None


@dataclass
class FCMResult:
    """Sorties complètes de l'algorithme FCM."""

    membership: np.ndarray
    centers: np.ndarray
    objective_history: list[float]
    n_iterations: int
    elapsed_time_sec: float
    converged: bool
    mpi_enabled: bool
    mpi_rank: int
    mpi_size: int


class FuzzyCMeans:
    """Algorithme Fuzzy C-Means codé manuellement."""

    def __init__(
        self,
        n_clusters: int = 4,
        m: float = 2.0,
        epsilon: float = 1e-3,
        max_iterations: int = 100,
        random_state: Optional[int] = 42,
        use_mpi: Optional[bool] = None,
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
        self.use_mpi = use_mpi

    @staticmethod
    def _resolve_mpi_mode(use_mpi: Optional[bool]) -> tuple[bool, int, int, object]:
        """Détermine si l'exécution MPI est activée."""
        if MPI is None:
            return False, 0, 1, None

        env_value = os.getenv("FCM_USE_MPI", "").strip().lower()
        env_requested = env_value in {"1", "true", "yes", "on"}
        requested = env_requested if use_mpi is None else use_mpi
        comm = MPI.COMM_WORLD
        size = int(comm.Get_size())
        rank = int(comm.Get_rank())
        enabled = bool(requested and size > 1)
        return enabled, rank, size, comm

    @staticmethod
    def _split_range(n_samples: int, rank: int, size: int) -> tuple[int, int]:
        """Découpe [0, n_samples) en blocs contigus équilibrés par rang MPI."""
        base = n_samples // size
        extra = n_samples % size
        start = rank * base + min(rank, extra)
        length = base + (1 if rank < extra else 0)
        return start, start + length

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
        mpi_enabled, mpi_rank, mpi_size, mpi_comm = self._resolve_mpi_mode(self.use_mpi)

        if mpi_enabled:
            start_idx, end_idx = self._split_range(n_samples, mpi_rank, mpi_size)
            local_data = data[start_idx:end_idx]
            membership = self._initialize_membership(local_data.shape[0])
        else:
            local_data = data
            membership = self._initialize_membership(n_samples)

        objective_history: list[float] = []
        converged = False
        start_time = time.perf_counter()

        for iteration in range(1, self.max_iterations + 1):
            if mpi_enabled:
                um = membership ** self.m
                local_numerator = um.T @ local_data
                local_denominator = np.sum(um, axis=0)[:, np.newaxis]
                numerator = mpi_comm.allreduce(local_numerator, op=MPI.SUM)
                denominator = mpi_comm.allreduce(local_denominator, op=MPI.SUM)
                denominator = np.clip(denominator, 1e-12, None)
                centers = numerator / denominator
            else:
                centers = self._update_centers(local_data, membership)

            distances_sq = self._euclidean_distances_squared(local_data, centers)
            membership_new = self._update_membership(distances_sq)
            objective_local = self._objective_function(membership_new, distances_sq)
            if mpi_enabled:
                objective_value = float(mpi_comm.allreduce(objective_local, op=MPI.SUM))
            else:
                objective_value = objective_local
            objective_history.append(objective_value)

            if callback is not None and (not mpi_enabled or mpi_rank == 0):
                callback(iteration, objective_value)

            # Critère de convergence:
            # ||U^(t+1) - U^(t)|| < epsilon
            delta_u_local = float(np.sum((membership_new - membership) ** 2))
            if mpi_enabled:
                delta_u = float(np.sqrt(mpi_comm.allreduce(delta_u_local, op=MPI.SUM)))
            else:
                delta_u = float(np.sqrt(delta_u_local))
            membership = membership_new

            if delta_u < self.epsilon:
                converged = True
                break

        elapsed = time.perf_counter() - start_time
        if mpi_enabled:
            elapsed = float(mpi_comm.allreduce(elapsed, op=MPI.MAX))
            gathered_membership = mpi_comm.allgather(membership)
            membership_full = np.vstack(gathered_membership) if gathered_membership else membership
        else:
            membership_full = membership

        return FCMResult(
            membership=membership_full,
            centers=centers,
            objective_history=objective_history,
            n_iterations=len(objective_history),
            elapsed_time_sec=elapsed,
            converged=converged,
            mpi_enabled=mpi_enabled,
            mpi_rank=mpi_rank,
            mpi_size=mpi_size,
        )
