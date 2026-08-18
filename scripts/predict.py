import argparse
import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.preprocessing import preprocess_gong_halpha
from src.postprocessing import postprocess_filament_instances
from src.utils_rle import export_rle_submission

def parse_args():
    parser = argparse.ArgumentParser(description="Solar Filament Segmentation Predictor")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with 2048x2048 images")
    parser.add_argument("--output_csv", type=str, default="submission/submission.csv", help="Output path")
    parser.add_argument("--score_thresh", type=float, default=0.45, help="Confidence threshold")
    parser.add_argument("--min_area", type=int, default=150, help="Minimum pixel area for filaments")
    return parser.parse_args()

def main():
    args = parse_args()
    image_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.jpg")) +
                         glob.glob(os.path.join(args.input_dir, "*.jpeg")))

    print(f"Found {len(image_paths)} images to process in {args.input_dir}")
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    predictions = {}
    for img_path in tqdm(image_paths, desc="Processing"):
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        tensor, disk_mask = preprocess_gong_halpha(img_path)

        # Baseline demo: placeholder empty predictions
        predictions[img_id] = []

    export_rle_submission(predictions, args.output_csv)

if __name__ == "__main__":
    main()
