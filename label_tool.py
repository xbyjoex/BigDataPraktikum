"""Interaktives Label-Tool für das Review-Set.

Zeigt die Patches aus outputs/manual_labels_template.csv nacheinander an
(links das komplette Schichtbild mit Markierung, rechts der Patch vergrößert)
und speichert jedes Label sofort nach outputs/manual_labels.csv. Ein
abgebrochener Durchlauf wird beim nächsten Start an der ersten
unbeschrifteten Stelle fortgesetzt. Die Modell-Scores werden bewusst nicht
angezeigt, damit sie das manuelle Urteil nicht beeinflussen.

Bedienung:
    1       Patch als anomal labeln und weiter
    0       Patch als regulär labeln und weiter
    Pfeile  vor/zurück (z. B. um ein Label zu korrigieren)
    q       beenden (Zwischenstand ist bereits gespeichert)

Start: .venv/bin/python label_tool.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

TEMPLATE_PATH = Path("outputs/manual_labels_template.csv")
PREDICTIONS_PATH = Path("outputs/resnet_predictions.csv")
OUTPUT_PATH = Path("outputs/manual_labels.csv")
SAVE_COLUMNS = ["sample_id", "image_path", "layer", "position", "anomaly_score", "resnet_prob_anomaly", "manual_label"]


def load_review_set() -> pd.DataFrame:
    source = OUTPUT_PATH if OUTPUT_PATH.exists() else TEMPLATE_PATH
    if not source.exists():
        raise FileNotFoundError(f"{TEMPLATE_PATH} fehlt. Bitte zuerst resnet_evaluation.ipynb ausführen.")

    review = pd.read_csv(source)
    if "manual_label" not in review:
        review["manual_label"] = pd.NA
    review["manual_label"] = review["manual_label"].astype("Int64")

    # Crop-Koordinaten aus den ResNet-Vorhersagen dazuholen
    predictions = pd.read_csv(PREDICTIONS_PATH)
    review = review.merge(predictions[["sample_id", "center_y", "center_x", "crop_side"]], on="sample_id", how="left")
    if review["crop_side"].isna().any():
        raise ValueError("Nicht alle Review-Patches in resnet_predictions.csv gefunden. Bitte resnet_evaluation.ipynb neu ausführen.")
    return review


class LabelTool:
    def __init__(self, review: pd.DataFrame):
        self.review = review
        unlabeled = np.flatnonzero(review["manual_label"].isna())
        self.index = int(unlabeled[0]) if len(unlabeled) else 0

        self.fig, (self.ax_full, self.ax_patch) = plt.subplots(1, 2, figsize=(14, 6), width_ratios=[2, 1])
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.show()

    def show(self):
        row = self.review.iloc[self.index]
        arr = np.asarray(Image.open(row["image_path"]).convert("L"))
        half = int(row["crop_side"]) // 2
        cy, cx = int(row["center_y"]), int(row["center_x"])
        patch = arr[max(0, cy - half):cy + half, max(0, cx - half):cx + half]

        self.ax_full.clear()
        self.ax_full.imshow(arr, cmap="gray")
        self.ax_full.add_patch(plt.Rectangle((cx - half, cy - half), 2 * half, 2 * half, fill=False, edgecolor="red", linewidth=1.5))
        self.ax_full.set_title(Path(row["image_path"]).name)
        self.ax_full.axis("off")

        self.ax_patch.clear()
        self.ax_patch.imshow(patch, cmap="gray")
        self.ax_patch.set_title(f"Layer {row['layer']} | Pos. {row['position']}")
        self.ax_patch.axis("off")

        label = self.review["manual_label"].iloc[self.index]
        label_text = "unbeschriftet" if pd.isna(label) else ("gelabelt: anomal" if label == 1 else "gelabelt: regulär")
        n_done = int(self.review["manual_label"].notna().sum())
        self.fig.suptitle(
            f"Patch {self.index + 1}/{len(self.review)} | {label_text} | gesamt: {n_done}/{len(self.review)}\n"
            "1 = anomal, 0 = regulär, Pfeile = vor/zurück, q = beenden"
        )
        self.fig.canvas.draw_idle()

    def save(self):
        self.review[SAVE_COLUMNS].to_csv(OUTPUT_PATH, index=False)

    def on_key(self, event):
        if event.key in ("0", "1"):
            self.review.iloc[self.index, self.review.columns.get_loc("manual_label")] = int(event.key)
            self.save()
            remaining = np.flatnonzero(self.review["manual_label"].isna())
            if len(remaining):
                nach_vorne = remaining[remaining > self.index]
                self.index = int(nach_vorne[0]) if len(nach_vorne) else int(remaining[0])
            elif self.index < len(self.review) - 1:
                self.index += 1
            self.show()
            if not len(remaining):
                self.fig.suptitle(f"Fertig: alle {len(self.review)} Patches gelabelt – q zum Beenden")
                self.fig.canvas.draw_idle()
        elif event.key == "left" and self.index > 0:
            self.index -= 1
            self.show()
        elif event.key == "right" and self.index < len(self.review) - 1:
            self.index += 1
            self.show()
        elif event.key == "q":
            n_done = int(self.review["manual_label"].notna().sum())
            print(f"Gespeichert: {OUTPUT_PATH} ({n_done}/{len(self.review)} gelabelt)")
            plt.close(self.fig)


if __name__ == "__main__":
    tool = LabelTool(load_review_set())
    plt.show()
    labeled = tool.review["manual_label"].dropna()
    print(f"{len(labeled)}/{len(tool.review)} gelabelt, davon anomal: {int(labeled.sum())}")
    if len(labeled) == len(tool.review):
        print("Zum Auswerten die letzte Zelle in resnet_evaluation.ipynb erneut ausführen.")
