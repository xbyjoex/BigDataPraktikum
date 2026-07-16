"""Erweitert das manuelle Label-Set um zusätzliche Patches.

Füllt outputs/manual_labels.csv bis zur Zielgröße auf, bestehende Labels
bleiben unverändert. Die neuen Patches setzen sich aus drei Gruppen zusammen:

- Grenzfälle: die höchsten Anomaliescores unter den regulär gelabelten
  Patches (knapp unter der Otsu-Schwelle, mögliche False Negatives).
- Widersprüche: Patches, bei denen ResNet-Vorhersage und Pseudo-Label
  nicht übereinstimmen.
- Zufallsauswahl aus dem Rest für eine unverzerrte Stichprobe.

Danach einfach label_tool.py starten. Es macht beim ersten unbeschrifteten
Patch weiter.

Aufruf: .venv/bin/python extend_review_set.py [zielgroesse]   (Default: 150)
"""

import sys
from pathlib import Path

import pandas as pd

from label_tool import OUTPUT_PATH, PREDICTIONS_PATH, SAVE_COLUMNS, TEMPLATE_PATH

SEED = 42
N_GRENZFAELLE = 20
N_WIDERSPRUECHE = 12


def main(target_size: int):
    source = OUTPUT_PATH if OUTPUT_PATH.exists() else TEMPLATE_PATH
    review = pd.read_csv(source)
    review["manual_label"] = review["manual_label"].astype("Int64")
    if len(review) >= target_size:
        print(f"Bereits {len(review)} Patches im Set, Ziel {target_size} – nichts zu tun.")
        return

    predictions = pd.read_csv(PREDICTIONS_PATH)
    candidates = predictions[~predictions["sample_id"].isin(review["sample_id"])]

    grenzfaelle = candidates[candidates["pseudo_label"] == 0].nlargest(N_GRENZFAELLE, "anomaly_score")
    candidates = candidates.drop(index=grenzfaelle.index)

    widersprueche = candidates[candidates["resnet_pred"] != candidates["pseudo_label"]]
    widersprueche = widersprueche.sample(n=min(N_WIDERSPRUECHE, len(widersprueche)), random_state=SEED)
    candidates = candidates.drop(index=widersprueche.index)

    n_random = target_size - len(review) - len(grenzfaelle) - len(widersprueche)
    zufall = candidates.sample(n=max(n_random, 0), random_state=SEED)

    neu = pd.concat([grenzfaelle, widersprueche, zufall]).sort_values(["layer", "position"])
    neu = neu[[c for c in SAVE_COLUMNS if c != "manual_label"]].copy()
    neu["manual_label"] = pd.NA

    erweitert = pd.concat([review[SAVE_COLUMNS], neu], ignore_index=True)
    erweitert.to_csv(OUTPUT_PATH, index=False)
    print(f"Set von {len(review)} auf {len(erweitert)} Patches erweitert:")
    print(f"  Grenzfälle: {len(grenzfaelle)}, Widersprüche: {len(widersprueche)}, Zufall: {len(zufall)}")
    print(f"Weiter geht es mit: .venv/bin/python label_tool.py")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 150)
