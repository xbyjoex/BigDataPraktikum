"""Interaktiver Grad-CAM-Browser über das Review-Set.

Zeigt zu jedem Patch das Original und die Grad-CAM-Heatmaps der drei
Trainingsphasen (Initialisierung, Mitte, Ende) des Fine-Tuning-Laufs.
Interessante Patches lassen sich markieren; die Markierungen landen in
outputs/gradcam_selection.csv und werden vom Notebook (Abschnitt 14)
als Abbildung gerendert.

Bedienung:
    m       Patch markieren/entmarkieren
    Pfeile  vor/zurück
    q       beenden (Markierungen sind bereits gespeichert)

Voraussetzung: resnet_evaluation.ipynb wurde mit Checkpoint-Speicherung
ausgeführt (outputs/checkpoints/fine_tuning_*.pt).

Start: .venv/bin/python gradcam_tool.py
"""

from pathlib import Path

import pandas as pd
import torch
import matplotlib.pyplot as plt

from gradcam import GradCAM, load_resnet18_checkpoint, load_patch, patch_to_tensor
from label_tool import load_review_set

CHECKPOINT_DIR = Path("outputs/checkpoints")
SELECTION_PATH = Path("outputs/gradcam_selection.csv")
PHASES = ["initialisierung", "mitte", "ende"]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


class GradCamBrowser:
    def __init__(self, review: pd.DataFrame):
        self.review = review.reset_index(drop=True)
        self.cams = {
            phase: GradCAM(load_resnet18_checkpoint(CHECKPOINT_DIR / f"fine_tuning_{phase}.pt", DEVICE))
            for phase in PHASES
        }
        self.marked = set(pd.read_csv(SELECTION_PATH)["sample_id"]) if SELECTION_PATH.exists() else set()
        self.index = 0
        self.cache = {}

        self.fig, self.axes = plt.subplots(1, len(PHASES) + 1, figsize=(16, 4.6))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.show()

    def compute(self, index: int):
        if index not in self.cache:
            row = self.review.iloc[index]
            patch = load_patch(row)
            x = patch_to_tensor(patch, DEVICE)
            self.cache[index] = (patch, {phase: cam(x, class_idx=1) for phase, cam in self.cams.items()})
        return self.cache[index]

    def show(self):
        row = self.review.iloc[self.index]
        patch, cam_results = self.compute(self.index)

        self.axes[0].clear()
        self.axes[0].imshow(patch, cmap="gray")
        label = row["manual_label"]
        label_text = "ohne Label" if pd.isna(label) else ("manuell: anomal" if label == 1 else "manuell: regulär")
        self.axes[0].set_title(f"Layer {row['layer']} | Pos. {row['position']}\n{label_text}")
        self.axes[0].axis("off")

        for ax, phase in zip(self.axes[1:], PHASES):
            cam, prob = cam_results[phase]
            ax.clear()
            ax.imshow(patch, cmap="gray")
            ax.imshow(cam, cmap="magma", alpha=0.45)
            ax.set_title(f"{phase}\nP(anomal)={prob:.2f}")
            ax.axis("off")

        marker = "MARKIERT" if row["sample_id"] in self.marked else "nicht markiert"
        self.fig.suptitle(
            f"Patch {self.index + 1}/{len(self.review)} | {marker} | insgesamt markiert: {len(self.marked)}\n"
            "m = markieren/entmarkieren, Pfeile = vor/zurück, q = beenden"
        )
        self.fig.canvas.draw_idle()

    def save(self):
        rows = self.review[self.review["sample_id"].isin(self.marked)]
        rows[["sample_id", "layer", "position", "manual_label"]].to_csv(SELECTION_PATH, index=False)

    def on_key(self, event):
        if event.key == "m":
            sample_id = self.review["sample_id"].iloc[self.index]
            self.marked.symmetric_difference_update({sample_id})
            self.save()
            self.show()
        elif event.key == "left" and self.index > 0:
            self.index -= 1
            self.show()
        elif event.key == "right" and self.index < len(self.review) - 1:
            self.index += 1
            self.show()
        elif event.key == "q":
            print(f"Gespeichert: {SELECTION_PATH} ({len(self.marked)} markiert)")
            plt.close(self.fig)


if __name__ == "__main__":
    browser = GradCamBrowser(load_review_set())
    plt.show()
    print(f"{len(browser.marked)} Patches markiert. Zum Rendern Abschnitt 14 in resnet_evaluation.ipynb ausführen.")
