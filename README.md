# Solar Filament Segmentation Challenge 2026 (IEEE BigDataCup)

An automated computer vision and deep learning framework for high-precision instance segmentation of solar filaments from GONG H-Alpha observations, directly optimizing Panoptic Quality (PQ) and topological structural continuity.

---

## Method Overview

1. **Preprocessing:**
   - Solar limb detection via geometric circle fitting.
   - 2D radial limb-darkening background profile correction.
   - Multi-channel representation: `[Flattened Intensity, CLAHE, Multi-Scale Frangi Vesselness]`.
2. **Architecture:**
   - **Mask2Former** panoptic transformer with a Swin Transformer / ConvNeXt backbone.
   - Auxiliary spine and centerline prediction head.
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

### 2. Run Prediction
```bash
python scripts/predict.py --input_dir /path/to/images --output_csv submission/submission.csv
```

---

## Project Structure
```
solar-filament-segmentation-2026/
├── configs/
├── notebooks/
├── src/
│   ├── models/
│   │   └── losses.py
│   ├── preprocessing.py
│   ├── postprocessing.py
│   └── utils_rle.py
├── scripts/
│   └── predict.py
├── submission/
├── weights/
├── requirements.txt
└── README.md
```
