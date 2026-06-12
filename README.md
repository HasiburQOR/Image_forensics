# SynthID Forensics Toolkit

Tools for analyzing and removing Google's SynthID watermark from AI-generated images — with forensic comparison, DCT-based perturbation, and automated PDF reporting.

## Tools

### `synthid_forensics.py` — Forensic Comparator

Compares your processed image (still detected) against a working tool's output (not detected) and generates a detailed PDF report covering:

- File & EXIF metadata differences
- Pixel-level statistics (mean diff, PSNR, per-channel)
- Color channel histograms (R/G/B)
- Frequency domain (FFT) analysis — radial profiles & magnitude maps
- DCT coefficient analysis (low/mid/high frequency bands)
- Noise floor comparison
- Entropy & GLCM texture metrics
- Visual difference heatmap
- Automated findings & recommendations on what to change

```bash
python synthid_forensics.py --a your_output.jpg --b working_tool_output.jpg
python synthid_forensics.py --a your_output.jpg --b working_tool_output.jpg --report analysis.pdf
```

### `synthid_working_tool.py` — SynthID Remover (Advanced)

Block-wise 8x8 DCT mid-frequency perturbation with aspect-ratio-aware profiles derived from forensic reports. Pipeline:

1. Classify aspect ratio (square/landscape/portrait)
2. Resize to reference dimensions (LANCZOS)
3. Per-channel 8x8 block DCT perturbation (mid-freq coefficients only)
4. Calibrated Gaussian noise injection
5. Subtle per-channel color shift
6. JPEG output (quality 88–92)

```bash
python synthid_working_tool.py input.jpg output.jpg
python synthid_working_tool.py input.jpg output.jpg --profile landscape
```

### `synthid_remover.py` — SynthID Remover (Simple)

Lightweight version using YCbCr Y-channel DCT attack + controlled noise:

```bash
python synthid_remover.py --input image.jpg --output clean.jpg --strength 1.5
```

### `resize.py` — Image Resizer

Quick resize utility for preparing comparison inputs:

```bash
python resize.py input.jpg 1440
```

## Requirements

```
Pillow
numpy
opencv-python
scipy
scikit-image
matplotlib
reportlab
```

Install with:

```bash
pip install -r requirements.txt
```

## Workflow

1. Run your own processing on a SynthID-marked image → `your_output.jpg`
2. Run `synthid_working_tool.py` on the same image → `reference_output.jpg`
3. Compare both with `synthid_forensics.py` → `forensic_report.pdf`
4. Read the "Recommendations" section — it tells you exactly which frequency bands, DCT coefficients, and noise levels to adjust
5. Tune your pipeline and repeat
# Image_forensics
# Image_forensics
