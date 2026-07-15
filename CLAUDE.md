# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

University project — **Big-Data Praktikum, Uni Leipzig, Topic 06**: *"Pseudo labelling und Vorhersage von Anomalien im 3D-Druck"*. Team: Jonas Paul & Lucas Berger. Advisor: B. Uhrich (uhrich@informatik.uni-leipzig.de).

**Goal.** Detect faulty layers in Selective Laser Melting (SLM) 3D printing from per-layer camera images. The pipeline has two ML stages:

1. **Convolutional Autoencoder** — projects each layer image into a low-dimensional code vector. Anomalous layers are derived from this latent space (pseudo-labelling step).
2. **Pretrained ResNet** — uses the pseudo-labels to learn a *geometry-independent* anomaly classifier that generalises to unseen part geometries.

Then evaluate + visualise correlations/anomalies across parts.

The motivation is that today SLM quality is only verified post-build via destructive tensile/compression tests — slow and expensive. In-process image analysis aims to flag defective layers during the build.

## Milestones (Testate)

All three must be passed; missing a deadline voids prior work.

| # | Deliverable | Deadline |
|---|---|---|
| 1 | Conceptual design doc (~4 pages): architecture, workflow, goals | end of May 2026 |
| 2 | Working implementation + evaluation + visualisations | mid/end July 2026 |
| 3 | 10 min presentation + 5 min discussion | early August or end September 2026 |

Testat 1 is done (design doc: `BigData (3).pdf`). The project is now in the **Testat 2 phase** — working implementation + evaluation + visualisations, due mid/end July 2026.

## Repository state

Implementation lives in three Jupyter notebooks (committed **with** outputs):

- `main.ipynb` — preprocessing (temporal-change calibration of the 9 part positions → 128×128 patches), Conv-Autoencoder training, anomaly score (α·MSE + (1−α)·latent distance, per-position latent reference, percentile-clipped normalisation), Otsu threshold → pseudo-labels, sanity check against known anomaly layer 715, t-SNE/PCA latent-space visualisations. Writes `outputs/pseudo_labels.csv` (the interface to the next stage) and figures.
- `resnet_evaluation.ipynb` — depends on `outputs/pseudo_labels.csv`. ResNet-18 transfer-learning comparison (feature extractor vs. fine-tuning), metrics per split, layer/position analyses, leave-one-position-out transfer test, manual-label template + evaluation.
- `explortionPicture.ipynb` — one-off EXIF/format exploration of the JPEGs.
- `label_tool.py` / `extend_review_set.py` — interactive manual labelling of the review set (keys 1/0, saves to `outputs/manual_labels.csv`) and seeded extension of that set (borderline cases, pseudo/ResNet disagreements, randoms).
- `gradcam.py` / `gradcam_tool.py` — Grad-CAM module (shared by notebook section 14) and interactive browser over the review set showing heatmaps for three fine-tuning phases (init/mitte/ende checkpoints from `outputs/checkpoints/`); marked patches land in `outputs/gradcam_selection.csv` and are rendered in the notebook.
- `BigData (3).pdf` — the Testat-1 design doc (NeurIPS style, German). The implementation deliberately deviates from it in two documented places: per-position latent reference instead of global mean (Gl. 2), and percentile-clipped normalisation instead of min-max. Fig. 3 in the PDF says τ95 but §2.4 and the code use Otsu — fix the figure for the final report.
- `README.md` — project plan in German + a `Fragenkatalog` of open questions for the advisor. Most questions are still *unanswered*; treat the answers as unknown; flag assumptions explicitly rather than silently picking one.
- `Autoencoder 3D Druck.md` — the official task description from the advisor (German).
- `BigDataPraktikum.md` — the general Praktikum rules and topic list. Git-ignored on purpose; do not modify or stage.
- `recoating/` — **1081 grayscale JPEGs**, 1624×1236, one per print layer. Filenames `YYYY-MM-DD_HH-MM-SS_layer_NNNNN.jpg`. These are the camera frames of the powder bed after recoating, before the next laser pass — the primary input for the autoencoder.
- `2021-02-18 Slm280_381_21/raw/` — 814 `.pkl` files, one per timestamp (~30s cadence). Contents unverified; likely per-frame sensor/temperature snapshots from the same build.
- `2021-02-18 Slm280_381_21/Result/` — aggregated temperature dataframe (`.pkl` + `.xlsx`) for the full build.

All data is from **one build run** (`Slm280_381_21`, 2021-02-18). Whether more builds / geometries will be supplied is one of the open questions for the advisor.

## Working conventions for this project

- **Language: German.** Documentation, the design doc, comments in user-facing artefacts, and the final presentation are all in German. Match that when editing those files. Code identifiers can stay English.
- **Don't answer the open questions yourself.** The `Fragenkatalog` in `README.md` (dataset scope, anomaly definition, ResNet variant, metric, framework, GPU, repo location, design-doc template) is meant to be resolved with the advisor. If a task depends on one of those answers, surface the assumption rather than baking a guess into code or the design doc.
- **Framework: PyTorch** (settled in the design doc §2.5, together with scikit-learn, NumPy, matplotlib).
- **Data is large and binary.** `recoating/` was unfortunately already committed (~300 MB history); do not make it worse — never `git add` the `2021-02-18 …/` pickle folders or `outputs/`.
- **Environment & running.** `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt`. Notebooks run headless via `.venv/bin/python -m nbconvert --to notebook --execute --inplace <notebook>.ipynb` (order: `main.ipynb` first, it produces `outputs/pseudo_labels.csv`). `resnet_evaluation.ipynb` supports env vars for quick validation: `FAST_DEV_RUN=1` (512 samples, 1 epoch) and `OUTPUT_DIR=...` to redirect outputs. Training uses CUDA → MPS → CPU in that order; on the Macs it runs on MPS.
- **No test suite exists.** Validation is done by executing the notebooks (FAST_DEV_RUN for the ResNet one) and checking the sanity-check cell for layer 715.

## What "anomaly" likely means (working hypothesis)

Not formally defined yet (Frage 2). Visually plausible defect signals in recoating images include: short-feed / streaks from the recoater blade, powder-bed waves, missing-powder patches, part shifting, prior-layer warpage poking through. The autoencoder is expected to flag these via either reconstruction error or distance in latent space — confirm with the advisor which signal is the intended one (Frage 3).
