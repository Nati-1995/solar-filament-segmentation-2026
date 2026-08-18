"""MAGFiLO (COCO-style) dataset loader for GONG H-alpha filaments.

Handles the two things that are specific to this competition:
  * multiple annotators per physical image -- the same file_name appears under
    several image entries (different batch-prefixed ids). We pool all of their
    annotations per file_name.
  * spines -- rasterised into an auxiliary target for the spine head / clDice.

Each item yields the 3-channel preprocessed input plus a foreground mask, a
spine mask, and (for validation) the per-annotator-union instance list.
"""
import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import preprocess_gong_halpha

NATIVE = 2048


def _poly_to_mask(seg, h=NATIVE, w=NATIVE):
    m = np.zeros((h, w), np.uint8)
    if not seg:
        return m
    poly = seg[0] if isinstance(seg[0], (list, tuple)) else seg
    pts = np.array(poly, np.float32).reshape(-1, 2)
    if len(pts) >= 3:
        cv2.fillPoly(m, [pts.round().astype(np.int32)], 1)
    return m


def _spine_to_mask(spine, h=NATIVE, w=NATIVE, thickness=7):
    m = np.zeros((h, w), np.uint8)
    if spine and len(spine) >= 4:
        pts = np.array(spine, np.float32).reshape(-1, 2).round().astype(np.int32)
        cv2.polylines(m, [pts], isClosed=False, color=1, thickness=thickness)
    return m


class MagfiloDataset(Dataset):
    def __init__(self, ann_json, images_dir, img_size=512, return_instances=False,
                 file_names=None):
        with open(ann_json) as f:
            coco = json.load(f)
        id2file = {im["id"]: im["file_name"] for im in coco["images"]}
        self.by_file = {}
        for a in coco["annotations"]:
            fn = id2file.get(a["image_id"])
            if fn is None:
                continue
            self.by_file.setdefault(fn, []).append(a)
        self.files = sorted(file_names) if file_names else sorted(self.by_file)
        self.images_dir = images_dir
        self.img_size = img_size
        self.return_instances = return_instances

    def group_key(self, file_name):
        """YYYYMMDD prefix -- used for leakage-free grouped CV splits."""
        stem = os.path.splitext(file_name)[0]
        return stem[:8]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        fn = self.files[i]
        anns = self.by_file[fn]
        tensor, disk_mask = preprocess_gong_halpha(os.path.join(self.images_dir, fn))

        fg = np.zeros((NATIVE, NATIVE), np.uint8)
        spine = np.zeros((NATIVE, NATIVE), np.uint8)
        for a in anns:
            fg |= _poly_to_mask(a.get("segmentation", []))
            spine |= _spine_to_mask(a.get("spine", []))

        instances = []
        if self.return_instances:
            # Fuse annotators: one GT instance per connected filament, not one
            # per annotator (raw union would double-count overlapping labels and
            # inflate FN). STAPLE consensus is the upgrade over this proxy.
            from skimage.morphology import label as _lbl
            lab = _lbl(fg)
            for k in range(1, int(lab.max()) + 1):
                instances.append((lab == k).astype(np.uint8))

        s = self.img_size
        x = cv2.resize(tensor, (s, s), interpolation=cv2.INTER_AREA)
        fg_s = cv2.resize(fg, (s, s), interpolation=cv2.INTER_NEAREST)
        sp_s = cv2.resize(spine, (s, s), interpolation=cv2.INTER_NEAREST)

        x = torch.from_numpy(x).permute(2, 0, 1).float()
        fg_t = torch.from_numpy(fg_s).unsqueeze(0).float()
        sp_t = torch.from_numpy(sp_s).unsqueeze(0).float()
        out = {"image": x, "fg": fg_t, "spine": sp_t, "file_name": fn}
        if self.return_instances:
            out["instances"] = instances  # native-res binary masks
        return out
