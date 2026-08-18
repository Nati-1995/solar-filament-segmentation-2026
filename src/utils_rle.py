import pandas as pd
import numpy as np
from pycocotools import mask as mask_util

def export_rle_submission(predictions: dict[str, list[np.ndarray]], output_path: str):
    """
    Converts binary instance masks to unquoted pycocotools RLE CSV.
    """
    records = []
    for img_id, masks in predictions.items():
        for k, m in enumerate(masks, start=1):
            filament_id = f"{img_id}_{k}"
            fortran_mask = np.asfortranarray(m.astype(np.uint8))
            rle_dict = mask_util.encode(fortran_mask)
            rle_str = rle_dict["counts"].decode("utf-8")
            records.append({
                "filament_id": filament_id,
                "segmentation_rle": rle_str
            })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} filament instances to {output_path}")
