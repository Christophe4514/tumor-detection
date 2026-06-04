"""Utilitaires d'entrée/sortie image et conversion pour Tkinter."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def load_image_bgr(path: str) -> np.ndarray:
    """Charge une image IRM à partir d'un chemin disque."""
    image_path = Path(path)
    if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError("Format non supporté. Utilisez PNG, JPG ou JPEG.")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Impossible de lire l'image. Vérifiez le chemin/fichier.")
    return image


def ensure_directory(path: str | Path) -> Path:
    """Crée le dossier s'il n'existe pas."""
    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_image(path: str | Path, image: np.ndarray) -> None:
    """Sauvegarde une image via OpenCV."""
    out_path = Path(path)
    ensure_directory(out_path.parent)
    success = cv2.imwrite(str(out_path), image)
    if not success:
        raise ValueError(f"Echec de sauvegarde: {out_path}")


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Convertit BGR (OpenCV) vers RGB (Pillow/Tkinter)."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def grayscale_to_rgb(image_gray: np.ndarray) -> np.ndarray:
    """Convertit grayscale vers RGB pour affichage uniforme."""
    return cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)


def normalize_float01_to_uint8(image_float: np.ndarray) -> np.ndarray:
    """Transforme une image float [0, 1] en uint8 [0, 255]."""
    return np.clip(image_float * 255.0, 0, 255).astype(np.uint8)


def to_tk_image(image_rgb: np.ndarray, max_size: Optional[tuple[int, int]] = (320, 320)) -> ImageTk.PhotoImage:
    """Convertit un tableau RGB NumPy en image Tkinter redimensionnée."""
    pil_image = Image.fromarray(image_rgb)

    if max_size is not None:
        if hasattr(Image, "Resampling"):
            pil_image.thumbnail(max_size, Image.Resampling.LANCZOS)
        else:
            pil_image.thumbnail(max_size, Image.LANCZOS)

    return ImageTk.PhotoImage(pil_image)
