import numpy as np
from skimage.morphology import remove_small_objects, binary_closing, disk

def postprocess_filament_instances(
    pred_masks: list[np.ndarray],
    pred_scores: list[float],
    min_area: int = 150,
    closing_radius: int = 3,
    score_threshold: float = 0.45,
    nms_iou_threshold: float = 0.35,
) -> list[np.ndarray]:
    """
    Refines instance masks to maximize Panoptic Quality (removes small false positives,
    bridges small discontinuities, resolves overlap).
    """
    refined_masks = []
    sorted_indices = np.argsort(pred_scores)[::-1]

    for idx in sorted_indices:
        if pred_scores[idx] < score_threshold:
            continue

        mask = pred_masks[idx].astype(bool)

        if closing_radius > 0:
            mask = binary_closing(mask, disk(closing_radius))

        mask = remove_small_objects(mask, min_size=min_area)
        if mask.sum() == 0:
            continue

        keep = True
        for existing in refined_masks:
            intersection = np.logical_and(mask, existing).sum()
            union = np.logical_or(mask, existing).sum()
            if (intersection / (union + 1e-8)) > nms_iou_threshold:
                keep = False
                break

        if keep:
            refined_masks.append(mask.astype(np.uint8))

    return refined_masks
