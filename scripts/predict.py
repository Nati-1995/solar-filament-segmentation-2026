"""Run the trained U-Net + spine model over a directory of test images and write
the RLE submission CSV. Optional rotational TTA (the Sun has no preferred
orientation, so rotations are label-preserving)."""
import os, sys, glob, argparse
import numpy as np
import torch
import cv2
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import preprocess_gong_halpha
from src.postprocessing import postprocess_filament_instances
from src.instancing import instances_from_semantic
from src.utils_rle import export_rle_submission
from src.models.unet import UNetSpine

NATIVE = 2048


def infer_maps(model, x, device, tta=True):
    """x: (1,3,H,W). Returns seg, spine prob maps at model resolution."""
    angles = [0, 90, 180, 270] if tta else [0]
    seg_acc, sp_acc = 0.0, 0.0
    for k in angles:
        xr = torch.rot90(x, k // 90, dims=(2, 3))
        with torch.no_grad():
            o = model(xr.to(device))
        seg = torch.rot90(torch.sigmoid(o["seg"]), -(k // 90), dims=(2, 3))
        sp = torch.rot90(torch.sigmoid(o["spine"]), -(k // 90), dims=(2, 3))
        seg_acc = seg_acc + seg; sp_acc = sp_acc + sp
    n = len(angles)
    return (seg_acc / n)[0, 0].cpu().numpy(), (sp_acc / n)[0, 0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_csv", default="submission/submission.csv")
    ap.add_argument("--weights", default="weights/unet_spine.pth")
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--score_thresh", type=float, default=0.45)
    ap.add_argument("--min_area", type=int, default=150)
    ap.add_argument("--no_tta", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNetSpine().to(device).eval()
    if os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print(f"loaded weights: {args.weights}")
    else:
        print(f"WARNING: no weights at {args.weights} -- predictions will be empty. Train first.")

    paths = sorted(glob.glob(os.path.join(args.input_dir, "*.jpg")) +
                   glob.glob(os.path.join(args.input_dir, "*.jpeg")))
    print(f"found {len(paths)} images in {args.input_dir}")
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    predictions = {}
    for p in tqdm(paths, desc="predict"):
        img_id = os.path.splitext(os.path.basename(p))[0]
        tensor, disk_mask = preprocess_gong_halpha(p)
        if not os.path.exists(args.weights):
            predictions[img_id] = []
            continue
        s = args.img_size
        x = cv2.resize(tensor, (s, s), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()
        seg, sp = infer_maps(model, x, device, tta=not args.no_tta)
        seg = cv2.resize(seg, (NATIVE, NATIVE)) * (disk_mask > 0)
        sp = cv2.resize(sp, (NATIVE, NATIVE)) * (disk_mask > 0)
        masks, scores = instances_from_semantic(
            seg, sp, min_area=args.min_area)
        masks = postprocess_filament_instances(
            masks, scores, min_area=args.min_area,
            score_threshold=args.score_thresh)
        predictions[img_id] = masks

    export_rle_submission(predictions, args.output_csv)


if __name__ == "__main__":
    main()
