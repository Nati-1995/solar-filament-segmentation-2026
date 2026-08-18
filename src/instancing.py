"""Turn a semantic foreground map into discrete filament instances.

This is the bridge between a semantic segmentation network and the
instance-level Panoptic Quality metric. Naive connected components merges
crossing filaments; naive splitting fragments barbed ones. We use
spine-guided marker-controlled watershed: the predicted spine (the dataset
ships spines, so we can supervise one) seeds one marker per filament, which
both prevents over-merge (separate seeds) and fragmentation (barbs flood back
to their parent spine).
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import (
    remove_small_objects, binary_closing, disk, skeletonize, label as sk_label,
)
from skimage.segmentation import watershed
from skimage.feature import peak_local_max


def instances_from_semantic(
    seg_prob,
    spine_prob=None,
    fg_thr=0.5,
    spine_thr=0.5,
    min_area=150,
    closing_radius=3,
    use_watershed=True,
):
    """seg_prob, spine_prob: float maps in [0,1] of equal shape. Returns
    (list_of_binary_masks, list_of_scores)."""
    fg = seg_prob >= fg_thr
    if closing_radius > 0:
        fg = binary_closing(fg, disk(closing_radius))
    fg = remove_small_objects(fg, min_size=min_area)
    if fg.sum() == 0:
        return [], []

    if not use_watershed:
        lbl = sk_label(fg)
        return _collect(lbl, seg_prob, min_area)

    # Markers: prefer the predicted spine skeleton, else distance-transform peaks.
    if spine_prob is not None and (spine_prob >= spine_thr).sum() > 0:
        seeds = skeletonize((spine_prob >= spine_thr) & fg)
        markers = sk_label(seeds)
        if markers.max() == 0:
            markers = _distance_markers(fg)
    else:
        markers = _distance_markers(fg)

    dist = ndi.distance_transform_edt(fg)
    lbl = watershed(-dist, markers, mask=fg)
    return _collect(lbl, seg_prob, min_area)


def _distance_markers(fg):
    dist = ndi.distance_transform_edt(fg)
    coords = peak_local_max(dist, min_distance=15, labels=fg)
    peaks = np.zeros(fg.shape, dtype=bool)
    if len(coords):
        peaks[tuple(coords.T)] = True
    return sk_label(peaks) if peaks.any() else sk_label(fg)


def _collect(lbl, seg_prob, min_area):
    masks, scores = [], []
    for k in range(1, int(lbl.max()) + 1):
        m = lbl == k
        if m.sum() < min_area:
            continue
        masks.append(m.astype(np.uint8))
        scores.append(float(seg_prob[m].mean()))
    return masks, scores
