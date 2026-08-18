# Solar Filament Segmentation Challenge 2026 (IEEE BigDataCup)

An automated computer vision and deep learning framework for high-precision instance segmentation of solar filaments from GONG H-Alpha observations, directly optimizing Panoptic Quality (PQ) and topological structural continuity.

---

## Method Overview

1. **Preprocessing:**
   - Solar limb detection via geometric circle fitting.
   - 2D radial limb-darkening background profile correction.
   - Multi-channel representation: `[Flattened Intensity, CLAHE, Multi-Scale Frangi Vesselness]`.
2. **Architecture:**
   - Baseline: compact **U-Net** with a shared encoder and two heads (filament foreground + spine/centerline). Robust on the small MAGFiLO set.
   - Stretch: **Mask2Former** panoptic transformer (Swin / ConvNeXt), reusing the same preprocessing, dataset, loss, PQ, and instancing code.
3. **Loss Function:**
   - Compound objective: L = L_BCE + L_Dice + lambda * L_clDice (centerline connectivity).
4. **Post-Processing:**
   - Small speckle noise suppression (Area < T_min).
   - Spine-guided bridging and Panoptic Quality threshold tuning.

---

## Quickstart & Reproduction

### 1. Installation
```bash
git clone https://github.com/Nati-1995/solar-filament-segmentation-2026.git
cd solar-filament-segmentation-2026
pip install -r requirements.txt
```

### 2. Train
```bash
python scripts/train.py \
  --ann_json data/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json \
  --images_dir data/train/train_images
```
Checkpoints are selected on **validation Panoptic Quality** (grouped by day to avoid leakage), saved to `weights/unet_spine.pth`.

### 3. Predict (writes RLE submission)
```bash
python scripts/predict.py --input_dir data/test/test_images --output_csv submission/submission.csv
```

> **Weights & reproducibility:** `.gitignore` excludes `weights/*.pth`. Attach the trained weights to a GitHub **Release** (or git-lfs) so the committee can reproduce predictions without extra files, as the rules require.

---

## Project Structure
```
solar-filament-segmentation-2026/
├── configs/
│   └── baseline.yaml
├── notebooks/
│   └── pipeline.ipynb        # full pipeline walkthrough (required deliverable)
├── src/
│   ├── preprocessing.py      # limb detect, flatten, CLAHE, Frangi -> 3ch
│   ├── dataset.py            # MAGFiLO COCO loader, multi-annotator pooling
│   ├── instancing.py         # spine-guided watershed: semantic -> instances
│   ├── metrics.py            # Panoptic Quality (SQ x RQ)
│   ├── postprocessing.py     # closing, min-area, IoU-NMS
│   ├── utils_rle.py          # masks -> pycocotools RLE submission
│   └── models/
│       ├── unet.py           # U-Net + spine head
│       └── losses.py         # BCE + Dice + clDice, combined loss
├── scripts/
│   ├── train.py              # grouped split, PQ-based checkpointing
│   └── predict.py            # inference + rotational TTA -> submission.csv
├── weights/
├── requirements.txt
└── README.md
```
