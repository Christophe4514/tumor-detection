"""Interface Tkinter de l'application de détection de tumeur par FCM."""

from __future__ import annotations

import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from core.dataset_calibration import run_calibration_and_evaluation
from core.fuzzy_classifier import FuzzyClassifierModel
from core.preprocessing import preprocess_mri
from core.segmentation import SegmentationResult, segment_with_fcm
from core.tumor_classification import TumorClassificationResult, classify_tumor_type
from core.tumor_detection import TumorDetectionResult, detect_tumor_from_segmentation
from utils.evaluation import format_confusion_matrix_text
from utils.image_utils import (
    bgr_to_rgb,
    ensure_directory,
    grayscale_to_rgb,
    load_image_bgr,
    save_image,
    to_tk_image,
)
from utils.metrics import RunMetrics, build_metrics


class BrainTumorFCMApp:
    """Application Tkinter principale."""

    DEFAULT_DATASET_DIR = Path("datasets") / "segmentation"
    DEFAULT_MODEL_PATH = Path("results") / "latest_fuzzy_classifier_model.json"

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Détection automatique des tumeurs cérébrales - FCM + OpenMPI")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 760)

        self._configure_style()
        self._build_layout()
        self._init_state()
        self.root.after(300, self._auto_initialize_calibration)

    def _configure_style(self) -> None:
        """Paramètres visuels ttk."""
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Title.TLabel", font=("Segoe UI", 13, "bold"))
        style.configure("Card.TLabelframe", padding=10)
        style.configure("Card.TLabelframe.Label", font=("Segoe UI", 10, "bold"))
        style.configure("Metric.TLabel", font=("Consolas", 10))

    def _build_layout(self) -> None:
        """Construit toute l'interface."""
        container = ttk.Frame(self.root, padding=10)
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(0, weight=3)
        container.columnconfigure(1, weight=2)
        container.rowconfigure(1, weight=1)

        title = ttk.Label(
            container,
            text="Détection automatique des tumeurs cérébrales sur IRM (FCM + OpenMPI)",
            style="Title.TLabel",
        )
        title.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        # Bloc affichage images
        self.viewer_frame = ttk.LabelFrame(container, text="Visualisation", style="Card.TLabelframe")
        self.viewer_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        self.viewer_frame.columnconfigure((0, 1), weight=1)
        self.viewer_frame.rowconfigure((0, 1), weight=1)

        self.image_labels: dict[str, ttk.Label] = {}
        self.image_refs: dict[str, object] = {}
        slots = [
            ("original", "Originale"),
            ("preprocessed", "Prétraitée"),
            ("segmented", "Segmentée"),
            ("tumor", "Tumeur extraite"),
        ]

        for index, (key, title_txt) in enumerate(slots):
            row = index // 2
            col = index % 2
            card = ttk.LabelFrame(self.viewer_frame, text=title_txt, style="Card.TLabelframe")
            card.grid(row=row, column=col, sticky="nsew", padx=5, pady=5)
            card.columnconfigure(0, weight=1)
            card.rowconfigure(0, weight=1)

            lbl = ttk.Label(card, text="Aucune image", anchor="center")
            lbl.grid(row=0, column=0, sticky="nsew")
            self.image_labels[key] = lbl

        # Bloc contrôle
        self.control_frame = ttk.LabelFrame(container, text="Contrôles & Paramètres", style="Card.TLabelframe")
        self.control_frame.grid(row=1, column=1, sticky="nsew")
        self.control_frame.columnconfigure(1, weight=1)

        self._build_control_widgets(self.control_frame)

        # Bloc journal + métriques
        bottom_frame = ttk.Frame(container)
        bottom_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        bottom_frame.columnconfigure(0, weight=2)
        bottom_frame.columnconfigure(1, weight=1)

        log_card = ttk.LabelFrame(bottom_frame, text="Évolution de la fonction coût Jm(U,V)", style="Card.TLabelframe")
        log_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        log_card.columnconfigure(0, weight=1)
        log_card.rowconfigure(0, weight=1)

        self.log_text = tk.Text(log_card, height=12, wrap="none", font=("Consolas", 9))
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.log_text.insert(tk.END, "Logs FCM...\n")
        self.log_text.configure(state=tk.DISABLED)

        metrics_card = ttk.LabelFrame(bottom_frame, text="Métriques", style="Card.TLabelframe")
        metrics_card.grid(row=0, column=1, sticky="nsew")
        metrics_card.columnconfigure(0, weight=1)

        self.metrics_labels = {
            "iterations": ttk.Label(metrics_card, text="Itérations: -", style="Metric.TLabel"),
            "time": ttk.Label(metrics_card, text="Temps (s): -", style="Metric.TLabel"),
            "objective": ttk.Label(metrics_card, text="Jm final: -", style="Metric.TLabel"),
            "convergence": ttk.Label(metrics_card, text="Convergence: -", style="Metric.TLabel"),
            "backend": ttk.Label(metrics_card, text="Backend FCM: -", style="Metric.TLabel"),
            "calib_acc": ttk.Label(metrics_card, text="Calibration Accuracy: -", style="Metric.TLabel"),
            "calib_f1": ttk.Label(metrics_card, text="Calibration F1 (macro): -", style="Metric.TLabel"),
            "calib_backend": ttk.Label(metrics_card, text="Calibration backend: -", style="Metric.TLabel"),
            "tumor_cluster": ttk.Label(metrics_card, text="Cluster tumoral: -", style="Metric.TLabel"),
            "tumor_type": ttk.Label(metrics_card, text="Type prédit: -", style="Metric.TLabel"),
            "calibration": ttk.Label(metrics_card, text="Calibration: non chargée", style="Metric.TLabel"),
        }
        for idx, label in enumerate(self.metrics_labels.values()):
            label.grid(row=idx, column=0, sticky="w", pady=2)

    def _build_control_widgets(self, parent: ttk.LabelFrame) -> None:
        """Construit les widgets de contrôle (boutons + paramètres FCM)."""
        row = 0

        # Paramètres FCM
        ttk.Label(parent, text="Nombre de clusters:").grid(row=row, column=0, sticky="w", pady=3)
        self.n_clusters_var = tk.IntVar(value=4)
        ttk.Spinbox(parent, from_=2, to=10, textvariable=self.n_clusters_var, width=8).grid(row=row, column=1, sticky="ew")
        row += 1

        ttk.Label(parent, text="m (flou):").grid(row=row, column=0, sticky="w", pady=3)
        self.m_var = tk.DoubleVar(value=2.0)
        ttk.Entry(parent, textvariable=self.m_var).grid(row=row, column=1, sticky="ew")
        row += 1

        ttk.Label(parent, text="epsilon:").grid(row=row, column=0, sticky="w", pady=3)
        self.epsilon_var = tk.DoubleVar(value=0.001)
        ttk.Entry(parent, textvariable=self.epsilon_var).grid(row=row, column=1, sticky="ew")
        row += 1

        ttk.Label(parent, text="max_iterations:").grid(row=row, column=0, sticky="w", pady=3)
        self.max_iter_var = tk.IntVar(value=100)
        ttk.Entry(parent, textvariable=self.max_iter_var).grid(row=row, column=1, sticky="ew")
        row += 1

        ttk.Label(parent, text="Calibration - max samples:").grid(row=row, column=0, sticky="w", pady=3)
        self.calibration_max_samples_var = tk.IntVar(value=120)
        ttk.Entry(parent, textvariable=self.calibration_max_samples_var).grid(row=row, column=1, sticky="ew")
        row += 1

        ttk.Separator(parent).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        row += 1

        # Boutons
        ttk.Button(parent, text="Charger image(s)", command=self.load_images).grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
        row += 1
        nav_frame = ttk.Frame(parent)
        nav_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
        nav_frame.columnconfigure((0, 1), weight=1)
        ttk.Button(nav_frame, text="Image précédente", command=self.load_previous_image).grid(row=0, column=0, sticky="ew", padx=(0, 2))
        ttk.Button(nav_frame, text="Image suivante", command=self.load_next_image).grid(row=0, column=1, sticky="ew", padx=(2, 0))
        row += 1
        self.batch_info_var = tk.StringVar(value="Lot: 0 image")
        ttk.Label(parent, textvariable=self.batch_info_var).grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 6))
        row += 1
        ttk.Button(parent, text="Prétraiter", command=self.run_preprocessing).grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
        row += 1
        ttk.Button(parent, text="Segmenter (FCM)", command=self.run_segmentation).grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
        row += 1
        ttk.Button(parent, text="Détecter tumeur", command=self.run_tumor_detection).grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
        row += 1
        ttk.Button(parent, text="Calibrer depuis datasets/segmentation", command=self.run_dataset_calibration).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=2
        )
        row += 1
        ttk.Button(parent, text="Sauvegarder résultat", command=self.save_results).grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
        row += 1
        ttk.Button(parent, text="Réinitialiser", command=self.reset_all).grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
        row += 1

        self.progress = ttk.Progressbar(parent, orient="horizontal", mode="determinate", maximum=100)
        self.progress.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        row += 1

        self.status_var = tk.StringVar(value="Prêt.")
        ttk.Label(parent, textvariable=self.status_var).grid(row=row, column=0, columnspan=2, sticky="w", pady=(5, 0))

    def _init_state(self) -> None:
        """Initialise les variables métier."""
        self.image_path: str | None = None
        self.image_paths: list[str] = []
        self.current_image_index: int = -1
        self.original_bgr: np.ndarray | None = None
        self.gray: np.ndarray | None = None
        self.preprocessed_uint8: np.ndarray | None = None
        self.preprocessed_normalized: np.ndarray | None = None
        self.segmentation_result: SegmentationResult | None = None
        self.tumor_result: TumorDetectionResult | None = None
        self.classification_result: TumorClassificationResult | None = None
        self.calibrated_model: FuzzyClassifierModel | None = None
        self.metrics: RunMetrics | None = None

    def _auto_initialize_calibration(self) -> None:
        """Charge le modèle calibré existant, sinon entraîne automatiquement depuis datasets/segmentation."""
        if self.DEFAULT_MODEL_PATH.exists():
            try:
                self.calibrated_model = FuzzyClassifierModel.load(self.DEFAULT_MODEL_PATH)
                self.metrics_labels["calibration"].configure(text="Calibration: modèle auto-chargé")
                self._set_status(f"Modèle calibré chargé: {self.DEFAULT_MODEL_PATH.name}")
                return
            except Exception:
                self.calibrated_model = None

        if self.DEFAULT_DATASET_DIR.exists():
            self.run_dataset_calibration(show_dialogs=False)

    def _clear_processing_views(self) -> None:
        """Nettoie les vues dérivées lorsqu'on change d'image source."""
        for key in ("preprocessed", "segmented", "tumor"):
            self.image_labels[key].configure(image="", text="Aucune image")
            self.image_refs.pop(key, None)

    def _invalidate_processing_state(self) -> None:
        """Invalide les étapes après chargement/changement d'image."""
        self.gray = None
        self.preprocessed_uint8 = None
        self.preprocessed_normalized = None
        self.segmentation_result = None
        self.tumor_result = None
        self.classification_result = None
        self.metrics = None
        self._clear_logs()
        self._clear_processing_views()
        self.metrics_labels["iterations"].configure(text="Itérations: -")
        self.metrics_labels["time"].configure(text="Temps (s): -")
        self.metrics_labels["objective"].configure(text="Jm final: -")
        self.metrics_labels["convergence"].configure(text="Convergence: -")
        self.metrics_labels["backend"].configure(text="Backend FCM: -")
        self.metrics_labels["calib_acc"].configure(text="Calibration Accuracy: -")
        self.metrics_labels["calib_f1"].configure(text="Calibration F1 (macro): -")
        self.metrics_labels["calib_backend"].configure(text="Calibration backend: -")
        self.metrics_labels["tumor_cluster"].configure(text="Cluster tumoral: -")
        self.metrics_labels["tumor_type"].configure(text="Type prédit: -")
        self.metrics_labels["calibration"].configure(
            text="Calibration: modèle chargé" if self.calibrated_model is not None else "Calibration: non chargée"
        )

    def _update_batch_info(self) -> None:
        """Met à jour l'indicateur du lot d'images."""
        if not self.image_paths or self.current_image_index < 0:
            self.batch_info_var.set("Lot: 0 image")
            return
        current = self.current_image_index + 1
        total = len(self.image_paths)
        current_name = Path(self.image_paths[self.current_image_index]).name
        self.batch_info_var.set(f"Lot: image {current}/{total} - {current_name}")

    def _load_current_image_from_batch(self) -> None:
        """Charge l'image actuellement sélectionnée dans le lot."""
        if not self.image_paths or self.current_image_index < 0:
            return

        selected = self.image_paths[self.current_image_index]
        self.original_bgr = load_image_bgr(selected)
        self.image_path = selected
        self._show_image("original", bgr_to_rgb(self.original_bgr))
        self._invalidate_processing_state()
        self._set_status(f"Image chargée: {Path(selected).name}")
        self._set_progress(10)
        self._update_batch_info()

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)
        self.root.update_idletasks()

    def _set_progress(self, value: float) -> None:
        self.progress["value"] = max(0.0, min(100.0, value))
        self.root.update_idletasks()

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _clear_logs(self) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.insert(tk.END, "Logs FCM...\n")
        self.log_text.configure(state=tk.DISABLED)

    def _show_image(self, key: str, image_rgb: np.ndarray) -> None:
        tk_img = to_tk_image(image_rgb, max_size=(450, 320))
        self.image_labels[key].configure(image=tk_img, text="")
        self.image_refs[key] = tk_img

    def _update_metrics_view(self) -> None:
        if self.metrics is None:
            return
        self.metrics_labels["iterations"].configure(text=f"Itérations: {self.metrics.iterations}")
        self.metrics_labels["time"].configure(text=f"Temps (s): {self.metrics.execution_time_sec:.4f}")
        self.metrics_labels["objective"].configure(text=f"Jm final: {self.metrics.final_objective_value:.6f}")
        self.metrics_labels["convergence"].configure(text=f"Convergence: {'Oui' if self.metrics.converged else 'Non'}")
        if self.segmentation_result is not None:
            fcm_meta = self.segmentation_result.fcm_result
            if fcm_meta.mpi_enabled:
                backend = f"OpenMPI ({fcm_meta.mpi_size} processus)"
            else:
                backend = "NumPy mono-processus"
            self.metrics_labels["backend"].configure(text=f"Backend FCM: {backend}")

    def load_images(self) -> None:
        """Charge une ou plusieurs IRM (PNG/JPG/JPEG)."""
        filetypes = [("Images", "*.png *.jpg *.jpeg"), ("PNG", "*.png"), ("JPG", "*.jpg *.jpeg")]
        selected = filedialog.askopenfilenames(title="Choisir une ou plusieurs images IRM", filetypes=filetypes)
        if not selected:
            return

        try:
            self.image_paths = list(selected)
            self.current_image_index = 0
            self._load_current_image_from_batch()
        except Exception as exc:
            messagebox.showerror("Erreur chargement", str(exc))

    def load_previous_image(self) -> None:
        """Charge l'image précédente du lot."""
        if not self.image_paths:
            messagebox.showinfo("Navigation", "Chargez d'abord une ou plusieurs images.")
            return
        if self.current_image_index <= 0:
            messagebox.showinfo("Navigation", "Vous êtes déjà sur la première image.")
            return

        try:
            self.current_image_index -= 1
            self._load_current_image_from_batch()
        except Exception as exc:
            messagebox.showerror("Erreur navigation", str(exc))

    def load_next_image(self) -> None:
        """Charge l'image suivante du lot."""
        if not self.image_paths:
            messagebox.showinfo("Navigation", "Chargez d'abord une ou plusieurs images.")
            return
        if self.current_image_index >= len(self.image_paths) - 1:
            messagebox.showinfo("Navigation", "Vous êtes déjà sur la dernière image.")
            return

        try:
            self.current_image_index += 1
            self._load_current_image_from_batch()
        except Exception as exc:
            messagebox.showerror("Erreur navigation", str(exc))

    def run_preprocessing(self) -> None:
        """Exécute le prétraitement."""
        if self.original_bgr is None:
            messagebox.showwarning("Prétraitement", "Veuillez d'abord charger une image.")
            return

        self._set_status("Prétraitement en cours...")
        self._set_progress(20)

        try:
            preproc = preprocess_mri(self.original_bgr, median_kernel_size=5)
            self.gray = preproc.gray_image
            self.preprocessed_uint8 = preproc.median_filtered
            self.preprocessed_normalized = preproc.normalized_image

            self._show_image("preprocessed", grayscale_to_rgb(self.preprocessed_uint8))
            self._set_progress(40)
            self._set_status("Prétraitement terminé.")
        except Exception as exc:
            self._set_status("Erreur pendant le prétraitement.")
            messagebox.showerror("Erreur prétraitement", f"Le prétraitement a échoué.\nDétail: {exc}")

    def run_segmentation(self) -> None:
        """Lance FCM et construit l'image segmentée."""
        if self.preprocessed_normalized is None:
            messagebox.showwarning("Segmentation", "Veuillez prétraiter l'image avant.")
            return

        n_clusters = self.n_clusters_var.get()
        m = self.m_var.get()
        epsilon = self.epsilon_var.get()
        max_iterations = self.max_iter_var.get()
        self._clear_logs()
        self._set_status("Segmentation FCM en cours...")
        self._set_progress(45)

        log_every = 2

        def per_iteration_callback(iter_idx: int, objective_value: float) -> None:
            progress_local = 45 + (iter_idx / max_iterations) * 40
            self._set_progress(progress_local)
            if iter_idx % log_every == 0 or iter_idx == 1 or iter_idx == max_iterations:
                self._append_log(f"Iter {iter_idx:03d} | Jm(U,V) = {objective_value:.8f}")
                # update() garde l'interface réactive pendant un calcul long.
                self.root.update()

        try:
            self.segmentation_result = segment_with_fcm(
                normalized_image=self.preprocessed_normalized,
                n_clusters=n_clusters,
                m=m,
                epsilon=epsilon,
                max_iterations=max_iterations,
                random_state=42,
                use_mpi=None,
                callback=per_iteration_callback,
            )

            self._show_image("segmented", grayscale_to_rgb(self.segmentation_result.segmented_image_uint8))

            fcm = self.segmentation_result.fcm_result
            self.metrics = build_metrics(
                iterations=fcm.n_iterations,
                execution_time_sec=fcm.elapsed_time_sec,
                objective_history=fcm.objective_history,
                converged=fcm.converged,
            )
            self._update_metrics_view()
            self._set_progress(88)
            self._set_status("Segmentation terminée.")
        except Exception as exc:
            messagebox.showerror("Erreur segmentation", str(exc))

    def run_tumor_detection(self) -> None:
        """Identifie automatiquement le cluster tumoral puis post-traite le masque."""
        if self.segmentation_result is None or self.preprocessed_uint8 is None:
            messagebox.showwarning("Détection", "Veuillez d'abord effectuer la segmentation.")
            return

        self._set_status("Détection tumorale en cours...")
        self._set_progress(90)

        try:
            self.tumor_result = detect_tumor_from_segmentation(
                segmentation_result=self.segmentation_result,
                preprocessed_gray_uint8=self.preprocessed_uint8,
                kernel_size=3,
                iterations=1,
            )
            self.classification_result = classify_tumor_type(
                refined_mask=self.tumor_result.refined_mask,
                preprocessed_gray_uint8=self.preprocessed_uint8,
                calibrated_model=self.calibrated_model,
                image_path=self.image_path,
            )

            self._show_image("tumor", grayscale_to_rgb(self.tumor_result.extracted_tumor))
            self.metrics_labels["tumor_cluster"].configure(
                text=(
                    f"Cluster tumoral: {self.tumor_result.tumor_cluster_index} "
                    f"(score={self.tumor_result.cluster_scores[self.tumor_result.tumor_cluster_index]:.3f})"
                )
            )
            self.metrics_labels["tumor_type"].configure(
                text=(
                    f"Type prédit: {self.classification_result.label} "
                    f"({self.classification_result.confidence:.2f})"
                )
            )
            self._set_progress(100)
            self._set_status(
                f"Détection terminée - classe prédite: {self.classification_result.label}"
            )
            messagebox.showinfo(
                "Classification tumorale",
                (
                    f"Classe prédite: {self.classification_result.label}\n"
                    f"Confiance: {self.classification_result.confidence:.2f}\n"
                    f"Méthode: {self.classification_result.method}\n\n"
                    f"Explication: {self.classification_result.explanation}"
                ),
            )

            # Fenêtre synthèse 6 vues demandées dans le cahier des charges.
            self._open_six_views_window()
        except Exception as exc:
            messagebox.showerror("Erreur détection", str(exc))

    def run_dataset_calibration(self, show_dialogs: bool = True) -> None:
        """Calibre un modèle flou supervisé puis affiche son évaluation."""
        dataset_dir = self.DEFAULT_DATASET_DIR
        if not dataset_dir.exists():
            if show_dialogs:
                messagebox.showerror(
                    "Dataset introuvable",
                    f"Dossier attendu introuvable:\n{dataset_dir}\n\nAjoutez vos données dans ce dossier.",
                )
            self._set_status("Calibration impossible: datasets/segmentation introuvable.")
            return

        self._set_status(f"Calibration/évaluation en cours depuis {dataset_dir} ...")
        self._set_progress(0)
        self._append_log(f"=== Calibration supervisée du classifieur flou ({dataset_dir}) ===")

        def progress_callback(current: int, total: int, msg: str) -> None:
            self._set_progress((current / max(1, total)) * 100.0)
            if current == 1 or current % 10 == 0 or current == total:
                self._append_log(f"[{current}/{total}] {msg}")
                self.root.update()

        try:
            result = run_calibration_and_evaluation(
                dataset_dir=dataset_dir,
                n_clusters=self.n_clusters_var.get(),
                m=self.m_var.get(),
                epsilon=self.epsilon_var.get(),
                max_iterations=min(self.max_iter_var.get(), 80),
                test_ratio=0.25,
                random_state=42,
                max_samples=max(20, int(self.calibration_max_samples_var.get())),
                progress_callback=progress_callback,
            )

            self.calibrated_model = result.model
            self.metrics_labels["calibration"].configure(
                text=f"Calibration: acc={result.report.accuracy:.3f} (test={result.n_test})"
            )
            self.metrics_labels["calib_acc"].configure(
                text=f"Calibration Accuracy: {result.report.accuracy:.3f}"
            )
            self.metrics_labels["calib_f1"].configure(
                text=f"Calibration F1 (macro): {result.report.macro_f1:.3f}"
            )
            calib_backend = (
                f"OpenMPI ({result.mpi_max_processes} processus)"
                if result.mpi_parallel_images > 0
                else "NumPy mono-processus"
            )
            self.metrics_labels["calib_backend"].configure(text=f"Calibration backend: {calib_backend}")

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_dir = ensure_directory(Path("results") / f"calibration_{timestamp}")
            model_path = output_dir / "fuzzy_classifier_model.json"
            report_path = output_dir / "evaluation_report.txt"
            self.calibrated_model.save(model_path)
            self.calibrated_model.save(self.DEFAULT_MODEL_PATH)
            report_text = format_confusion_matrix_text(result.report)
            report_path.write_text(report_text, encoding="utf-8")

            self._append_log(report_text)
            self._append_log(
                (
                    f"Temps calibration: {result.calibration_time_sec:.2f}s | "
                    f"Backend: {calib_backend} | "
                    f"Images traitées en parallèle: {result.mpi_parallel_images}/{result.n_total}"
                )
            )
            self._set_status(f"Calibration terminée. Accuracy test={result.report.accuracy:.3f}")
            self._set_progress(100)
            if show_dialogs:
                messagebox.showinfo(
                    "Calibration terminée",
                    (
                        f"Accuracy test: {result.report.accuracy:.4f}\n"
                        f"Macro F1-score: {result.report.macro_f1:.4f}\n"
                        f"Weighted F1-score: {result.report.weighted_f1:.4f}\n"
                        f"Train/Test: {result.n_train}/{result.n_test}\n"
                        f"Temps calibration: {result.calibration_time_sec:.2f}s\n"
                        f"Backend: {calib_backend}\n"
                        f"Images en mode parallèle: {result.mpi_parallel_images}/{result.n_total}\n"
                        f"Images ignorées: {result.skipped_files}\n\n"
                        f"Modèle: {model_path}\n"
                        f"Modèle courant: {self.DEFAULT_MODEL_PATH}\n"
                        f"Rapport: {report_path}"
                    ),
                )
        except Exception as exc:
            self._set_status("Erreur calibration/évaluation.")
            if show_dialogs:
                messagebox.showerror("Erreur calibration/évaluation", str(exc))

    def _open_six_views_window(self) -> None:
        """Affiche simultanément 6 vues: originale, gris, prétraitée, segmentée, masque, tumeur."""
        if (
            self.original_bgr is None
            or self.gray is None
            or self.preprocessed_uint8 is None
            or self.segmentation_result is None
            or self.tumor_result is None
        ):
            return

        window = tk.Toplevel(self.root)
        window.title("Visualisation comparative (6 vues)")
        window.geometry("1200x700")

        items = [
            ("Image originale", bgr_to_rgb(self.original_bgr)),
            ("Niveaux de gris", grayscale_to_rgb(self.gray)),
            ("Image prétraitée", grayscale_to_rgb(self.preprocessed_uint8)),
            ("Image segmentée", grayscale_to_rgb(self.segmentation_result.segmented_image_uint8)),
            ("Masque tumoral", grayscale_to_rgb(self.tumor_result.refined_mask)),
            ("Tumeur extraite", grayscale_to_rgb(self.tumor_result.extracted_tumor)),
        ]

        image_refs: list[object] = []
        for idx, (title, arr) in enumerate(items):
            row = idx // 3
            col = idx % 3
            card = ttk.LabelFrame(window, text=title, padding=6)
            card.grid(row=row, column=col, sticky="nsew", padx=6, pady=6)
            window.columnconfigure(col, weight=1)
            window.rowconfigure(row, weight=1)

            tk_img = to_tk_image(arr, max_size=(360, 260))
            lbl = ttk.Label(card, image=tk_img)
            lbl.pack(fill=tk.BOTH, expand=True)
            image_refs.append(tk_img)

        # Empêche le garbage collector de supprimer les images Tkinter.
        window.image_refs = image_refs  # type: ignore[attr-defined]

    def save_results(self) -> None:
        """Sauvegarde les résultats dans le dossier results/."""
        if self.segmentation_result is None:
            messagebox.showwarning("Sauvegarde", "Aucun résultat de segmentation à sauvegarder.")
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = ensure_directory(Path("results") / f"run_{timestamp}")

        try:
            if self.original_bgr is not None:
                save_image(output_dir / "original.png", self.original_bgr)
            if self.gray is not None:
                save_image(output_dir / "gray.png", self.gray)
            if self.preprocessed_uint8 is not None:
                save_image(output_dir / "preprocessed.png", self.preprocessed_uint8)

            save_image(output_dir / "segmented.png", self.segmentation_result.segmented_image_uint8)

            for idx, mask in enumerate(self.segmentation_result.cluster_masks):
                save_image(output_dir / f"cluster_mask_{idx}.png", mask)

            if self.tumor_result is not None:
                save_image(output_dir / "tumor_initial_mask.png", self.tumor_result.initial_mask)
                save_image(output_dir / "tumor_refined_mask.png", self.tumor_result.refined_mask)
                save_image(output_dir / "tumor_extracted.png", self.tumor_result.extracted_tumor)

            metrics_path = output_dir / "metrics.txt"
            lines = []
            if self.metrics is not None:
                lines.append(f"iterations={self.metrics.iterations}")
                lines.append(f"execution_time_sec={self.metrics.execution_time_sec:.6f}")
                lines.append(f"final_objective_value={self.metrics.final_objective_value:.8f}")
                lines.append(f"converged={self.metrics.converged}")
            if self.tumor_result is not None:
                lines.append(f"tumor_cluster_index={self.tumor_result.tumor_cluster_index}")
            if self.classification_result is not None:
                lines.append(f"predicted_tumor_type={self.classification_result.label}")
                lines.append(f"classification_confidence={self.classification_result.confidence:.4f}")
                lines.append(f"classification_method={self.classification_result.method}")
                lines.append(f"classification_explanation={self.classification_result.explanation}")
            if self.calibrated_model is not None:
                lines.append("calibrated_model_loaded=True")

            metrics_path.write_text("\n".join(lines), encoding="utf-8")
            self._set_status(f"Résultats sauvegardés: {output_dir}")
            messagebox.showinfo("Sauvegarde", f"Résultats enregistrés dans:\n{output_dir}")
        except Exception as exc:
            messagebox.showerror("Erreur sauvegarde", str(exc))

    def reset_all(self) -> None:
        """Réinitialise l'application."""
        keep_model = self.calibrated_model
        self._init_state()
        self.calibrated_model = keep_model
        self._set_progress(0)
        self._set_status("Réinitialisé.")
        self._clear_logs()
        self._update_batch_info()

        for key, label in self.image_labels.items():
            label.configure(image="", text="Aucune image")
            self.image_refs.pop(key, None)

        self.metrics_labels["iterations"].configure(text="Itérations: -")
        self.metrics_labels["time"].configure(text="Temps (s): -")
        self.metrics_labels["objective"].configure(text="Jm final: -")
        self.metrics_labels["convergence"].configure(text="Convergence: -")
        self.metrics_labels["backend"].configure(text="Backend FCM: -")
        self.metrics_labels["calib_acc"].configure(text="Calibration Accuracy: -")
        self.metrics_labels["calib_f1"].configure(text="Calibration F1 (macro): -")
        self.metrics_labels["calib_backend"].configure(text="Calibration backend: -")
        self.metrics_labels["tumor_cluster"].configure(text="Cluster tumoral: -")
        self.metrics_labels["tumor_type"].configure(text="Type prédit: -")
        self.metrics_labels["calibration"].configure(
            text="Calibration: modèle chargé" if self.calibrated_model is not None else "Calibration: non chargée"
        )
