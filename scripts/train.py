"""Train the U-Net + spine baseline with a leakage-free grouped split
(images from the same day stay on the same side). Selects checkpoints on
validation Panoptic Quality, not on loss.
"""
import os, sys, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import MagfiloDataset, NATIVE
from src.models.unet import UNetSpine
from src.models.losses import CombinedLoss
from src.instancing import instances_from_semantic
from src.metrics import PQAccumulator
import cv2


def grouped_split(ds, val_frac=0.15, seed=42):
    groups = {}
    for fn in ds.files:
        groups.setdefault(ds.group_key(fn), []).append(fn)
    keys = sorted(groups)
    rng = np.random.default_rng(seed); rng.shuffle(keys)
    n_val = max(1, int(len(keys) * val_frac))
    val_keys = set(keys[:n_val])
    train_f = [fn for k in keys if k not in val_keys for fn in groups[k]]
    val_f = [fn for k in val_keys for fn in groups[k]]
    return train_f, val_f


@torch.no_grad()
def evaluate(model, ds_val, device, img_size):
    model.eval()
    acc = PQAccumulator()
    for item in ds_val:
        x = item["image"].unsqueeze(0).to(device)
        out = model(x)
        seg = torch.sigmoid(out["seg"])[0, 0].cpu().numpy()
        sp = torch.sigmoid(out["spine"])[0, 0].cpu().numpy()
        seg = cv2.resize(seg, (NATIVE, NATIVE))
        sp = cv2.resize(sp, (NATIVE, NATIVE))
        preds, _ = instances_from_semantic(seg, sp)
        acc.update(preds, item["instances"])
    return acc.compute()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann_json", required=True)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="weights/unet_spine.pth")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = MagfiloDataset(args.ann_json, args.images_dir, args.img_size)
    train_f, val_f = grouped_split(base)
    ds_tr = MagfiloDataset(args.ann_json, args.images_dir, args.img_size, file_names=train_f)
    ds_va = MagfiloDataset(args.ann_json, args.images_dir, args.img_size,
                           return_instances=True, file_names=val_f)
    dl = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True)

    model = UNetSpine().to(device)
    crit = CombinedLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best = -1.0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for ep in range(1, args.epochs + 1):
        model.train(); running = 0.0
        for b in dl:
            x = b["image"].to(device)
            tgt = {"fg": b["fg"].to(device), "spine": b["spine"].to(device)}
            opt.zero_grad()
            loss, _ = crit(model(x), tgt)
            loss.backward(); opt.step()
            running += float(loss)
        m = evaluate(model, ds_va, device, args.img_size)
        print(f"epoch {ep:03d}  loss {running/len(dl):.4f}  "
              f"val PQ {m['pq']:.4f} (SQ {m['sq']:.3f} RQ {m['rq']:.3f})")
        if m["pq"] > best:
            best = m["pq"]
            torch.save(model.state_dict(), args.out)
            print(f"  saved new best -> {args.out}")

    print(f"done. best val PQ = {best:.4f}")


if __name__ == "__main__":
    main()
