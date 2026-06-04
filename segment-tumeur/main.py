"""Point d'entrée de l'application de segmentation FCM."""

import tkinter as tk

from ui.interface import BrainTumorFCMApp


def main() -> None:
    root = tk.Tk()
    BrainTumorFCMApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
