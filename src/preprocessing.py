import cv2
import numpy as np
from skimage.filters import frangi

def preprocess_gong_halpha(image_path: str):
    """
    Loads a 2048x2048 GONG H-alpha image and generates a 3-channel normalized tensor:
    [Limb-flattened, CLAHE-enhanced, Frangi vesselness]
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
    h, w = img.shape

    # 1. Solar Limb Detection
    _, thresh = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_cnt = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(largest_cnt)
    cx, cy, radius = int(cx), int(cy), int(radius)

    disk_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(disk_mask, (cx, cy), radius - 4, 1, thickness=-1)

    # 2. Radial Limb-Darkening Flattening
    y_coords, x_coords = np.indices((h, w))
    r_dist = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2) / radius
    valid_pixels = (disk_mask == 1) & (r_dist < 0.98)
    r_vals = r_dist[valid_pixels]
    i_vals = img[valid_pixels]

    poly_coeffs = np.polyfit(r_vals, i_vals, deg=4)
    radial_profile = np.clip(np.polyval(poly_coeffs, r_dist), 10.0, 255.0)

    img_flat = np.where(disk_mask == 1, (img / radial_profile) * 128.0, 0.0)
    img_flat = np.clip(img_flat, 0, 255).astype(np.uint8)

    # 3. CLAHE Local Contrast Enhancement
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(16, 16))
    img_clahe = np.where(disk_mask == 1, clahe.apply(img_flat), 0)

    # 4. Frangi Vesselness Filter for Barbs & Spines
    inv_flat = np.where(disk_mask == 1, 255 - img_flat, 0)
    vesselness = frangi(inv_flat, sigmas=range(1, 6, 2), black_ridges=False)
    vesselness = (vesselness / (vesselness.max() + 1e-8) * 255).astype(np.uint8)
    vesselness = np.where(disk_mask == 1, vesselness, 0)

    # 3-channel normalized tensor
    tensor = np.stack([img_flat, img_clahe, vesselness], axis=-1).astype(np.float32) / 255.0
    return tensor, disk_mask
