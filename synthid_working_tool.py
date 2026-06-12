#!/usr/bin/env python3
"""
SynthID Removal Tool – Forensic Parameter Adaptation

Based on analysis reports comparing unsuccessful (A) vs successful (B) outputs.
Implements:
- Aspect‑ratio aware resizing (to reference dimensions from reports)
- Block‑wise DCT perturbation (mid‑frequency coefficients only)
- Calibrated Gaussian noise injection (std dev from successful runs)
- Slight per‑channel histogram shifts (colour balancing)
- JPEG output with quality 88‑92

Usage:
    python remove_synthid.py input.jpg output.jpg
"""

import argparse
import numpy as np
import cv2
from PIL import Image
import io
import sys

# ----------------------------------------------------------------------
# Configuration: aspect‑ratio profiles derived from forensic reports
# Keys: (category, target dimensions)
# Values: noise_std, dct_mid_strength, colour_shift_strength, jpeg_quality
# ----------------------------------------------------------------------
PROFILES = {
    "square": {          # 1:1 – from report 916 (1024x1024 -> 1440x1440)
        "target_size": (1440, 1440),
        "noise_std": 9.4,          # Similar to both A & B (~9.4)
        "dct_mid_strength": 12.0,  # Increase mid‑freq diff (target >14)
        "colour_shift": 0.02,      # Subtle channel shift
        "jpeg_quality": 90,
    },
    "landscape": {       # width > height – from report 169 (1376x768 -> 1600x899)
        "target_size": (1600, 899),
        "noise_std": 10.9,         # Theirs B noise floor ~10.9
        "dct_mid_strength": 14.0,  # Raise mid‑freq diff from 16.0 to ~20
        "colour_shift": 0.015,
        "jpeg_quality": 88,
    },
    "portrait": {        # height > width – from report 11 (1536x2752 -> 899x1600)
        "target_size": (899, 1600),
        "noise_std": 4.1,          # Theirs B noise std = 4.09 (much lower than A)
        "dct_mid_strength": 13.0,  # Increase mid‑freq diff (was 14.3)
        "colour_shift": 0.025,
        "jpeg_quality": 92,
    },
    "default": {         # fallback for other ratios (e.g. 4:3, 3:2)
        "target_size": (1280, 1280),  # conservative square
        "noise_std": 7.0,
        "dct_mid_strength": 10.0,
        "colour_shift": 0.01,
        "jpeg_quality": 90,
    }
}

def get_aspect_ratio_category(w, h, eps=0.05):
    """
    Classify image aspect ratio into square, landscape, or portrait.
    Square: |w/h - 1| < eps
    Landscape: w/h > 1 + eps
    Portrait: w/h < 1 - eps
    """
    ratio = w / h
    if abs(ratio - 1.0) < eps:
        return "square"
    elif ratio > 1.0 + eps:
        return "landscape"
    else:
        return "portrait"

def resize_to_target(img, target_size):
    """
    Resize image to target (width, height) using high‑quality interpolation.
    """
    return cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)

def dct_perturbation(img, strength, block_size=8):
    """
    Apply mid‑frequency DCT perturbation to each 8x8 block.
    - Avoids DC coefficient (index 0,0)
    - Targets indices where (row >= block_size//4 and row < block_size//2)
      and similarly for col (roughly h//8 to h//2 in block coordinates)
    - Adds Gaussian noise to selected DCT coefficients.
    """
    h, w = img.shape[:2]
    img_float = img.astype(np.float32)
    # Work on each channel independently
    for c in range(img.shape[2]):
        channel = img_float[:, :, c]
        # Process image in blocks
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                # Extract block (handle edges with full block)
                block = channel[i:min(i+block_size, h), j:min(j+block_size, w)]
                bh, bw = block.shape
                if bh < block_size or bw < block_size:
                    continue  # skip partial blocks for simplicity
                # DCT
                dct_block = cv2.dct(block)
                # Mid‑frequency mask: avoid DC, avoid highest frequencies
                # Indices from block_size//4 to block_size//2 (approx)
                start_idx = block_size // 4
                end_idx = block_size // 2
                for y in range(start_idx, end_idx):
                    for x in range(start_idx, end_idx):
                        # Add noise scaled by strength
                        noise = np.random.normal(0, strength / 10.0)
                        dct_block[y, x] += noise
                # Inverse DCT
                channel[i:i+block_size, j:j+block_size] = cv2.idct(dct_block)
        img_float[:, :, c] = channel
    # Clip to valid range and convert back to uint8
    img_float = np.clip(img_float, 0, 255)
    return img_float.astype(np.uint8)

def add_calibrated_noise(img, noise_std):
    """
    Add Gaussian noise with given standard deviation.
    Noise is added in YUV space to better match perceptual models (optional),
    but here we add directly in RGB for simplicity and consistency with reports.
    """
    noise = np.random.normal(0, noise_std, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def subtle_colour_shift(img, strength):
    """
    Slightly shift channel histograms as seen in successful outputs.
    strength: small float (0.01 to 0.03) – multiplies channel difference.
    """
    img_float = img.astype(np.float32)
    # Compute per‑channel mean shift
    mean_r = np.mean(img_float[:, :, 0])
    mean_g = np.mean(img_float[:, :, 1])
    mean_b = np.mean(img_float[:, :, 2])
    # Shift each channel away from the overall mean a tiny bit
    overall_mean = (mean_r + mean_g + mean_b) / 3.0
    img_float[:, :, 0] += strength * (overall_mean - mean_r)
    img_float[:, :, 1] += strength * (overall_mean - mean_g)
    img_float[:, :, 2] += strength * (overall_mean - mean_b)
    return np.clip(img_float, 0, 255).astype(np.uint8)

def process_image(input_path, output_path, profile_override=None):
    """
    Main processing pipeline:
    1. Load image
    2. Determine aspect ratio category
    3. Resize to target dimensions for that category
    4. Apply DCT mid‑frequency perturbation
    5. Add calibrated noise
    6. Apply colour shift
    7. Save as JPEG with appropriate quality
    """
    # Load image
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Cannot read image: {input_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB for consistent handling

    h, w = img.shape[:2]
    category = get_aspect_ratio_category(w, h)
    if profile_override:
        category = profile_override
    profile = PROFILES.get(category, PROFILES["default"])
    print(f"[INFO] Category: {category}, using profile: {profile}")

    # 1. Resize
    target_w, target_h = profile["target_size"]
    img_resized = resize_to_target(img, (target_w, target_h))
    print(f"[INFO] Resized from {w}x{h} to {target_w}x{target_h}")

    # 2. DCT perturbation (mid‑frequency only)
    img_dct = dct_perturbation(img_resized, profile["dct_mid_strength"])
    print(f"[INFO] Applied DCT mid‑frequency perturbation (strength={profile['dct_mid_strength']})")

    # 3. Add noise
    img_noisy = add_calibrated_noise(img_dct, profile["noise_std"])
    print(f"[INFO] Added Gaussian noise (std={profile['noise_std']})")

    # 4. Colour shift
    img_final = subtle_colour_shift(img_noisy, profile["colour_shift"])
    print(f"[INFO] Applied colour shift (strength={profile['colour_shift']})")

    # Convert back to BGR for OpenCV save
    img_final_bgr = cv2.cvtColor(img_final, cv2.COLOR_RGB2BGR)

    # 5. Save as JPEG
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), profile["jpeg_quality"]]
    success = cv2.imwrite(output_path, img_final_bgr, encode_params)
    if not success:
        raise RuntimeError(f"Failed to save image to {output_path}")
    print(f"[INFO] Saved to {output_path} (JPEG quality {profile['jpeg_quality']})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove SynthID watermark with forensic parameter adaptation")
    parser.add_argument("input", help="Path to input image")
    parser.add_argument("output", help="Path to output JPEG")
    parser.add_argument("--profile", choices=["square", "landscape", "portrait", "default"],
                        help="Override automatic aspect‑ratio detection")
    args = parser.parse_args()

    try:
        process_image(args.input, args.output, profile_override=args.profile)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)