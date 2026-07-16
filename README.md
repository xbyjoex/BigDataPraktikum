# Big-Data-Praktikum: Anomalieerkennung im 3D-Druck

Dieses Projekt analysiert Schichtbilder eines 3D-Druckprozesses mit neun Bauteilen pro Bild. 
Die Pipeline erkennt die Bauteile, erzeugt Patches, trainiert einen Autoencoder für Pseudo-Labelling und trainiert 
anschließend ein ResNet-18, um diese Pseudo-Labels zu reproduzieren und auszuwerten.

Dabei ist zu beachten, dass die generierten Label aus dem Autoencoder als Pseudo-Labels angesehen werden müssen. 
Sie sind keine manuell bestätigte Ground Truth und beschreiben standardmäßig den Zustand eines Bauteil-Patches in 
einer konkreten Schicht.

## Projektstruktur

Das Projekt unterteilt sich in folgende Dateien:
- `main.ipynb` </br>
Dient zur Vorverarbeitung, passendes Bauteil-Cropping, dem Autoencoder-Training und das Pseudo-Labelling. Zusätzlich 
wird eine Latentraumdiagnose durchgeführt. In ihr befindet sich auch eine `dataclass` namens `Config`, die alle 
Konfigurationen für das Notebook beinhaltet für eine schnelle Anpassung der Parameter.


- `resnet_evaluation.ipynb` </br>
In diesem Notebook passiert das ResNet-Training, dem Feature-Extractor und Fine-Tuning. Schlussendlich wird eine 
Evaluierung durchgeführt.


- `gradcam.py` </br>
Ist ein Interaktives Tool zum manuellen Labeln des Review-Sets, damit noch genauer gelernt oder evaluiert werden kann.


- `label_tool.py` </br>
Hilfsfunktionen für Grad-CAM.


- `extend_review_set.py` </br>
Damit kann das Review-Set manuell um weitere Grenzfälle ergänzt werden.


- `gradcam_tool.py` </br>
Interaktiver Browser zum Markieren interessanter Grad-CAM-Beispiele


- `requirements.txt` </br>
Beinhaltet die benutzten Python Bibliotheken. Diese können in einer virtuellen Umgebung mit dieser Datei einfach
installiert werden.


Die Bilddaten der Schichten sollten im Ordner `recoating/` gespeichert sein. Dabei ist es wichtig, dass die Dateinamen die Schichtnummer enthalten wie z. B. `..._layer_00715.jpg`.

## Pipeline

### 1. Autoencoder und Pseudo-Labelling

Notebook: `main.ipynb`

1. Alle JPEGs aus `recoating/` werden anhand der Layernummer sortiert.
2. Aus einer zeitlichen Änderungskarte werden die neun Bauteilzentren bestimmt.
3. Pro Schicht werden neun Patches (Patch pro Bauteil) erzeugt, also `1081 * 9 = 9729` Samples.
4. Ein Convolutional Autoencoder rekonstruiert die Patches.
5. Aus Rekonstruktionsfehler und Latentraum-Distanz wird ein Anomalie-Score gebildet.
6. Die Otsu-Schwelle erzeugt das `pseudo_label` in binärer Form.
7. Diagnoseplots prüfen Scores, Rekonstruktionen, bekannte Verdachtsschichten und Latentraumstruktur.

Zentrale Outputs in `outputs`-Ordner zur Auswertung oder Weiterverarbeitung:

- `outputs/pseudo_labels.csv`
- `outputs/conv_autoencoder.pt`
- `outputs/autoencoder_history.csv`
- `outputs/latent_tsne.csv`
- `outputs/figures/`

### 2. ResNet-Training und Evaluation

Notebook: `resnet_evaluation.ipynb`

1. `outputs/pseudo_labels.csv` wird geladen.
2. Die gleichen Bauteil-Crops werden anhand der gespeicherten Crop-Zentren rekonstruiert.
3. Der Split erfolgt layer-basiert, damit ähnliche Schichten nicht gleichzeitig in Training und Test liegen.
4. ResNet-18 wird als Feature-Extractor und als Fine-Tuning-Variante trainiert.
5. Die beste Strategie wird anhand der Validierungsmetriken ausgewählt.
6. Testmetriken, Vorhersagen, Score-Kurven, Heatmaps und Feature-Projektionen werden gespeichert.
7. Optional werden Leave-One-Position-Out, manuelle Labels und Grad-CAM ausgewertet.

Zentrale Outputs in `outputs`-Ordner zur Auswertung oder Weiterverarbeitung:

- `outputs/resnet_predictions.csv`
- `outputs/resnet_metrics.json`
- `outputs/resnet_classification_report.txt`
- `outputs/resnet_history_feature_extractor.csv`
- `outputs/resnet_history_fine_tuning.csv`
- `outputs/lopo_metrics.csv`
- `outputs/manual_labels_template.csv`
- `outputs/checkpoints/`
- `outputs/figures/`

## Installation

Empfohlen ist eine virtuelle Umgebung mit Python 3.12 oder 3.13.

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name bigdata-3d-druck
```

Für das ResNet-Notebook ist `torchvision` wichtig, weil darüber das ImageNet-vortrainierte ResNet-18 geladen wird. 
Falls `torchvision` oder die Gewichte nicht verfügbar sind, nutzt das Notebook einen lokalen ResNet-18-Fallback ohne 
ImageNet-Pretraining. Für die finale Auswertung sollte aber `torchvision` verwendet werden.

## Ausführung

Die Notebooks sollten in dieser Reihenfolge ausgeführt werden:

1. `main.ipynb`
2. `resnet_evaluation.ipynb`

Dann zuerst `main.ipynb` vollständig ausführen und danach `resnet_evaluation.ipynb`.

Alternativ per Kommandozeile:

```bash
.venv/bin/jupyter nbconvert --execute --to notebook --inplace main.ipynb
.venv/bin/jupyter nbconvert --execute --to notebook --inplace resnet_evaluation.ipynb
```

Für kurze Tests kann das ResNet-Notebook mit weniger Daten und nur einer Epoche laufen:

```bash
FAST_DEV_RUN=1 .venv/bin/jupyter nbconvert --execute --to notebook --inplace resnet_evaluation.ipynb
```

## Manuelle Kontrolle

Nach dem ResNet-Notebook existiert `outputs/manual_labels_template.csv`. Damit kann ein kleines Review-Set manuell 
gelabelt werden:

```bash
.venv/bin/python label_tool.py
```

Bedienung:

- `1`: Patch als anomal labeln
- `0`: Patch als regulär labeln
- Pfeiltasten: vor/zurück
- `q`: beenden

Das Tool schreibt nach `outputs/manual_labels.csv`. Danach die manuelle Label-Auswertungszelle in `resnet_evaluation.ipynb` 
erneut ausführen.

Das Review-Set kann erweitert werden:

```bash
.venv/bin/python extend_review_set.py 150
```

## Grad-CAM

Das ResNet-Notebook speichert während des Fine-Tuning-Laufs Checkpoints in `outputs/checkpoints/`. 
Damit können Grad-CAM-Heatmaps für verschiedene Trainingsphasen erzeugt werden.

Interaktiver Browser:

```bash
.venv/bin/python gradcam_tool.py
```

Bedienung:

- `m`: Patch markieren oder entmarkieren
- Pfeiltasten: vor/zurück
- `q`: beenden

Die Auswahl wird in `outputs/gradcam_selection.csv` gespeichert. Danach Abschnitt 14 in `resnet_evaluation.ipynb` erneut 
ausführen, um die finale Grad-CAM-Abbildung zu rendern.

## Typische Probleme

| Problem | Lösung |
|---|---|
| `outputs/pseudo_labels.csv` fehlt | Zuerst `main.ipynb` vollständig ausführen |
| ResNet läuft ohne Pretraining | `torchvision` installieren und Netzwerk/Cache für ImageNet-Gewichte prüfen |
| Grad-CAM-Checkpoints fehlen | `resnet_evaluation.ipynb` bis einschließlich Fine-Tuning erneut ausführen |
| Manuelle Label-Auswertung leer | Erst `label_tool.py` ausführen und `outputs/manual_labels.csv` speichern lassen |
| Lauf dauert sehr lange | Für Tests `FAST_DEV_RUN=1` oder `MAX_SAMPLES=512` verwenden |
