"""Panoptic Quality (PQ) for filament instance segmentation.

PQ = SQ * RQ  (Kirillov et al., CVPR 2019, doi 10.1109/CVPR.2019.00963)
  SQ = mean IoU over matched (TP) pairs
  RQ = TP / (TP + 0.5*FP + 0.5*FN)
A predicted instance matches a ground-truth instance iff their IoU > 0.5,
which guarantees a unique one-to-one match. Fragmentation (one GT hit by many
small preds) and over-merge (one pred covering many GTs) both fall below 0.5
and are penalised as FP/FN -- exactly what the competition scores.
"""
import numpy as np


def _iou(a, b):
    a = a.astype(bool); b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return inter / union


def match_instances(pred_masks, gt_masks, iou_thr=0.5):
    """Return (matched_pairs, ious, unmatched_pred_idx, unmatched_gt_idx)."""
    n_p, n_g = len(pred_masks), len(gt_masks)
    pairs = []
    for i in range(n_p):
        for j in range(n_g):
            v = _iou(pred_masks[i], gt_masks[j])
            if v > iou_thr:
                pairs.append((v, i, j))
    pairs.sort(reverse=True)  # greedy, high IoU first (unique at thr>0.5)
    used_p, used_g, matched, ious = set(), set(), [], []
    for v, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j)
        matched.append((i, j)); ious.append(v)
    unm_p = [i for i in range(n_p) if i not in used_p]
    unm_g = [j for j in range(n_g) if j not in used_g]
    return matched, ious, unm_p, unm_g


def panoptic_quality(pred_masks, gt_masks, iou_thr=0.5):
    if len(pred_masks) == 0 and len(gt_masks) == 0:
        return dict(pq=1.0, sq=1.0, rq=1.0, tp=0, fp=0, fn=0, iou_sum=0.0)
    matched, ious, unm_p, unm_g = match_instances(pred_masks, gt_masks, iou_thr)
    tp, fp, fn = len(matched), len(unm_p), len(unm_g)
    iou_sum = float(np.sum(ious))
    sq = iou_sum / tp if tp else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) else 0.0
    return dict(pq=sq * rq, sq=sq, rq=rq, tp=tp, fp=fp, fn=fn, iou_sum=iou_sum)


class PQAccumulator:
    """Micro-averaged PQ across a whole image set (accumulate then compute)."""
    def __init__(self, iou_thr=0.5):
        self.iou_thr = iou_thr
        self.tp = self.fp = self.fn = 0
        self.iou_sum = 0.0

    def update(self, pred_masks, gt_masks):
        m, ious, up, ug = match_instances(pred_masks, gt_masks, self.iou_thr)
        self.tp += len(m); self.fp += len(up); self.fn += len(ug)
        self.iou_sum += float(np.sum(ious))

    def compute(self):
        sq = self.iou_sum / self.tp if self.tp else 0.0
        rq = self.tp / (self.tp + 0.5 * self.fp + 0.5 * self.fn) if (self.tp + self.fp + self.fn) else 1.0
        return dict(pq=sq * rq, sq=sq, rq=rq, tp=self.tp, fp=self.fp, fn=self.fn)
