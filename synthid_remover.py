import cv2
import numpy as np
from PIL import Image
import os
from scipy.fft import dct, idct

def get_aspect_ratio_category(width, height):
    ratio = width / height
    if abs(ratio - 1.0) < 0.15:        # Square
        return "square", 1440, 1440, 9.35, 95
    elif ratio > 1.0:                   # Landscape (16:9)
        return "landscape", 1600, 899, 10.9, 92
    else:                               # Portrait (9:16)
        return "portrait", 899, 1600, 4.1, 92


def targeted_dct_attack(img_array, strength=1.5):
    """Targeted mid-frequency DCT perturbation in YCbCr space to avoid low-frequency and color distortion"""
    # Convert to YCrCb
    ycrcb = cv2.cvtColor(img_array, cv2.COLOR_RGB2YCrCb)
    y_channel = ycrcb[:, :, 0].astype(np.float32)
    h, w = y_channel.shape
    
    # DCT on Y channel
    dct_y = dct(dct(y_channel, axis=0, norm='ortho'), axis=1, norm='ortho')
    
    # Mid-frequency mask
    mask = np.zeros_like(dct_y)
    y_start, y_end = max(1, h//8), h//2
    x_start, x_end = max(1, w//8), w//2
    mask[y_start:y_end, x_start:x_end] = 1.0
    
    # Perturb mid-frequencies of Y channel
    noise = np.random.normal(0, strength * np.maximum(np.abs(dct_y), 1.0), dct_y.shape)
    dct_perturbed = dct_y + (noise * mask)
    
    # Inverse DCT
    reconstructed_y = idct(idct(dct_perturbed, axis=1, norm='ortho'), axis=0, norm='ortho')
    
    # Replace Y channel
    ycrcb_perturbed = ycrcb.copy()
    ycrcb_perturbed[:, :, 0] = np.clip(reconstructed_y, 0, 255).astype(np.uint8)
    
    # Convert back to RGB
    final = cv2.cvtColor(ycrcb_perturbed, cv2.COLOR_YCrCb2RGB)
    return final


def add_controlled_noise(img, noise_level=7.5):
    """Add subtle noise"""
    noise = np.random.normal(0, noise_level, img.shape).astype(np.int16)
    noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy


def process_image(input_path, output_path, strength=1.5):
    # Load image
    img = Image.open(input_path).convert("RGB")
    img_array = np.array(img)
    
    orig_h, orig_w = img_array.shape[:2]
    category, target_w, target_h, noise_level, jpeg_quality = get_aspect_ratio_category(orig_w, orig_h)
    
    print(f"Detected: {category.upper()} | Original: {orig_w}x{orig_h} -> Resizing to {target_w}x{target_h}")
    print(f"Parameters: Noise Level = {noise_level} | JPEG Quality = {jpeg_quality} | DCT Strength = {strength}")
    
    # Resize using high quality
    resized = cv2.resize(img_array, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Main SynthID attack (YCbCr Y-channel DCT)
    perturbed = targeted_dct_attack(resized, strength=strength)
    
    # Add target level noise
    final = add_controlled_noise(perturbed, noise_level=noise_level)
    
    # Save as JPEG with target quality
    cv2.imwrite(output_path, cv2.cvtColor(final, cv2.COLOR_RGB2BGR), 
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    
    print(f"SUCCESS Saved: {output_path} ({target_w}x{target_h})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SynthID Remover - Optimized Version")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", default="clean_output.jpg", help="Output path")
    parser.add_argument("--strength", type=float, default=1.5, help="Perturbation strength (1.0 ~ 2.5)")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    process_image(args.input, args.output, strength=args.strength)