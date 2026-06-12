"""
SynthID Forensic Comparator
============================
Give it two images:
  A = your processed image (SynthID still detected)
  B = the working tool's output (SynthID not detected)

It will produce a detailed PDF report + terminal printout covering:
  1. Basic file & metadata differences
  2. Pixel-level difference stats
  3. Color channel analysis (R/G/B histograms)
  4. Frequency domain (FFT) comparison — this is where SynthID lives
  5. DCT coefficient analysis
  6. Noise floor analysis
  7. Entropy & texture analysis
  8. Visual heatmaps (diff map, FFT map)
  9. Exact recommendations based on findings

Usage:
  python synthid_forensics.py --a your_output.jpg --b working_tool_output.jpg
  python synthid_forensics.py --a your_output.jpg --b working_tool_output.jpg --report report.pdf
"""

import argparse
import os
import io
import struct
import numpy as np
import cv2
from PIL import Image, ExifTags
from scipy.fft import dct, idct
from scipy import stats
from skimage.measure import shannon_entropy
from skimage.feature import graycomatrix, graycoprops
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Image as RLImage, Table, TableStyle,
                                 HRFlowable, KeepTogether)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

W, H = A4
styles = getSampleStyleSheet()

# ── Styles ─────────────────────────────────────────────
def make_styles():
    return {
        "title": ParagraphStyle("T", parent=styles["Title"], fontSize=18, leading=24,
                                textColor=colors.HexColor("#1a1a2e"), alignment=TA_CENTER, spaceAfter=4),
        "sub":   ParagraphStyle("Su", parent=styles["Normal"], fontSize=10,
                                textColor=colors.HexColor("#555"), alignment=TA_CENTER, spaceAfter=10),
        "h2":    ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12,
                                textColor=colors.HexColor("#1a1a2e"), spaceBefore=12, spaceAfter=5),
        "body":  ParagraphStyle("Bo", parent=styles["Normal"], fontSize=9, leading=13,
                                textColor=colors.HexColor("#333"), spaceAfter=6, alignment=TA_JUSTIFY),
        "mono":  ParagraphStyle("Mo", parent=styles["Normal"], fontSize=8, leading=11,
                                fontName="Courier", textColor=colors.HexColor("#222"), spaceAfter=3),
        "good":  ParagraphStyle("Go", parent=styles["Normal"], fontSize=9,
                                textColor=colors.HexColor("#1a7a1a"), fontName="Helvetica-Bold"),
        "bad":   ParagraphStyle("Ba", parent=styles["Normal"], fontSize=9,
                                textColor=colors.HexColor("#cc0000"), fontName="Helvetica-Bold"),
        "warn":  ParagraphStyle("Wa", parent=styles["Normal"], fontSize=9,
                                textColor=colors.HexColor("#b36200"), fontName="Helvetica-Bold"),
        "foot":  ParagraphStyle("Fo", parent=styles["Normal"], fontSize=7.5,
                                textColor=colors.HexColor("#999"), alignment=TA_CENTER),
    }

ST = make_styles()

# ── Helpers ────────────────────────────────────────────
def load(path):
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    gray = np.array(img.convert("L"), dtype=np.float32)
    return img, arr, gray

def fig_to_rl(fig, w_mm, h_mm):
    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", dpi=130, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return RLImage(buf, width=w_mm*mm, height=h_mm*mm)

def stat_table(rows, col_widths=None):
    if col_widths is None:
        col_widths = [55*mm, 55*mm, 55*mm]
    t = Table(rows, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0),  colors.HexColor("#dce8f5")),
        ("FONTNAME",      (0,0),(-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,-1), 8.5),
        ("TEXTCOLOR",     (0,0),(-1,-1), colors.HexColor("#1a1a2e")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.HexColor("#f7f7f7"), colors.white]),
        ("GRID",          (0,0),(-1,-1), 0.4, colors.HexColor("#cccccc")),
        ("LEFTPADDING",   (0,0),(-1,-1), 6),
        ("RIGHTPADDING",  (0,0),(-1,-1), 6),
        ("TOPPADDING",    (0,0),(-1,-1), 4),
        ("BOTTOMPADDING", (0,0),(-1,-1), 4),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
    ]))
    return t


# ══════════════════════════════════════════════════════
# ANALYSIS MODULES
# ══════════════════════════════════════════════════════

# 1. FILE & METADATA
def analyze_file(path_a, path_b):
    results = {}
    for label, path in [("yours", path_a), ("theirs", path_b)]:
        img = Image.open(path)
        size_kb = os.path.getsize(path) / 1024
        exif_raw = img._getexif() if hasattr(img, "_getexif") and img._getexif() else {}
        exif = {}
        if exif_raw:
            for tag_id, val in exif_raw.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                exif[tag] = str(val)[:80]
        results[label] = {
            "size_kb": round(size_kb, 2),
            "dimensions": f"{img.size[0]}x{img.size[1]}",
            "mode": img.mode,
            "format": img.format or path.split(".")[-1].upper(),
            "exif_keys": list(exif.keys())[:8],
            "exif": exif,
        }
    return results

# 2. PIXEL STATS
def analyze_pixels(arr_a, arr_b):
    # Resize b to match a if needed
    if arr_a.shape != arr_b.shape:
        h, w = arr_a.shape[:2]
        img_b = Image.fromarray(arr_b.astype(np.uint8))
        img_b = img_b.resize((w, h), Image.LANCZOS)
        arr_b = np.array(img_b, dtype=np.float32)

    diff = arr_a - arr_b
    results = {
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "max_diff":      float(np.max(np.abs(diff))),
        "std_diff":      float(np.std(diff)),
        "psnr":          float(10 * np.log10(255**2 / np.mean(diff**2))) if np.mean(diff**2) > 0 else 999,
        "diff_arr":      diff,
        "arr_b_matched": arr_b,
    }
    # Per channel
    for i, ch in enumerate(["R","G","B"]):
        d = diff[:,:,i]
        results[f"mean_{ch}"] = float(np.mean(d))
        results[f"std_{ch}"]  = float(np.std(d))
    return results

# 3. HISTOGRAM
def analyze_histogram(arr_a, arr_b_matched):
    fig, axes = plt.subplots(2, 3, figsize=(10, 5))
    ch_colors = ["red","green","blue"]
    labels = ["Yours (A)", "Theirs (B)"]
    for row, arr in enumerate([arr_a, arr_b_matched]):
        for col, (ch, color) in enumerate(zip(range(3), ch_colors)):
            axes[row][col].hist(arr[:,:,ch].ravel(), bins=128, color=color,
                                alpha=0.7, density=True)
            axes[row][col].set_title(f"{labels[row]} — {'RGB'[col]} channel", fontsize=8)
            axes[row][col].set_xlim(0,255)
            axes[row][col].tick_params(labelsize=7)
    plt.suptitle("Color Channel Histograms", fontsize=10, fontweight="bold")
    plt.tight_layout()
    return fig

# 4. FFT ANALYSIS — the most important one for SynthID
def analyze_fft(gray_a, gray_b):
    results = {}
    figs = []

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for idx, (label, gray) in enumerate([("Yours", gray_a), ("Theirs", gray_b)]):
        fft = np.fft.fft2(gray)
        fshift = np.fft.fftshift(fft)
        magnitude = np.log1p(np.abs(fshift))
        results[label] = {
            "magnitude": magnitude,
            "fshift": fshift,
            "mean_mag": float(np.mean(magnitude)),
            "std_mag":  float(np.std(magnitude)),
            "max_mag":  float(np.max(magnitude)),
        }
        axes[idx].imshow(magnitude, cmap="inferno")
        axes[idx].set_title(f"FFT Magnitude — {label}", fontsize=9)
        axes[idx].axis("off")

    # Difference FFT
    diff_mag = results["Yours"]["magnitude"] - results["Theirs"]["magnitude"]
    im = axes[2].imshow(diff_mag, cmap="RdBu_r", vmin=-3, vmax=3)
    axes[2].set_title("FFT Difference (Yours − Theirs)\nRed=yours stronger, Blue=theirs stronger", fontsize=8)
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    plt.suptitle("Frequency Domain (FFT) Analysis — SynthID lives here", fontsize=10, fontweight="bold")
    plt.tight_layout()
    figs.append(fig)

    # Radial frequency profile
    fig2, ax = plt.subplots(figsize=(8, 3.5))
    h, w = gray_a.shape
    cy, cx = h//2, w//2
    y_idx, x_idx = np.ogrid[:h, :w]
    dist = np.sqrt((y_idx-cy)**2 + (x_idx-cx)**2).astype(int)
    max_r = min(cy, cx)

    for label, gray in [("Yours (A)", gray_a), ("Theirs (B)", gray_b)]:
        fft = np.fft.fftshift(np.fft.fft2(gray))
        mag = np.log1p(np.abs(fft))
        profile = [np.mean(mag[dist == r]) for r in range(max_r)]
        ax.plot(profile, label=label, linewidth=1.5)

    ax.set_xlabel("Radial frequency (0=DC, right=high freq)", fontsize=8)
    ax.set_ylabel("Mean log magnitude", fontsize=8)
    ax.set_title("Radial Frequency Profile — divergence = watermark signal difference", fontsize=9)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    figs.append(fig2)

    results["mean_diff"] = float(np.mean(np.abs(diff_mag)))
    results["mid_freq_diff"] = float(np.mean(np.abs(diff_mag[
        cy//4:3*cy//4, cx//4:3*cx//4])))
    return results, figs

# 5. DCT ANALYSIS
def analyze_dct(gray_a, gray_b):
    h = min(gray_a.shape[0], gray_b.shape[0])
    w = min(gray_a.shape[1], gray_b.shape[1])
    ga = gray_a[:h,:w]
    gb = gray_b[:h,:w]

    dct_a = dct(dct(ga, axis=0, norm="ortho"), axis=1, norm="ortho")
    dct_b = dct(dct(gb, axis=0, norm="ortho"), axis=1, norm="ortho")
    diff  = dct_a - dct_b

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for i, (d, title) in enumerate([(dct_a,"DCT — Yours"),(dct_b,"DCT — Theirs"),(diff,"DCT Difference")]):
        clipped = np.clip(d, np.percentile(d,2), np.percentile(d,98))
        axes[i].imshow(clipped, cmap="viridis" if i<2 else "RdBu_r", aspect="auto")
        axes[i].set_title(title, fontsize=9)
        axes[i].axis("off")
    plt.suptitle("DCT Coefficient Analysis", fontsize=10, fontweight="bold")
    plt.tight_layout()

    return {
        "mean_abs_dct_diff": float(np.mean(np.abs(diff))),
        "mid_freq_dct_diff": float(np.mean(np.abs(diff[h//8:h//2, w//8:w//2]))),
        "high_freq_dct_diff":float(np.mean(np.abs(diff[h//2:, w//2:]))),
        "low_freq_dct_diff": float(np.mean(np.abs(diff[:h//8, :w//8]))),
    }, fig

# 6. NOISE FLOOR
def analyze_noise(gray_a, gray_b):
    def get_noise(gray):
        blurred = cv2.GaussianBlur(gray.astype(np.uint8), (5,5), 0).astype(np.float32)
        noise = gray - blurred
        return noise

    noise_a = get_noise(gray_a)
    noise_b = get_noise(gray_b)

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for i, (n, title) in enumerate([(noise_a,"Noise Floor — Yours"),
                                     (noise_b,"Noise Floor — Theirs"),
                                     (noise_a - noise_b,"Noise Difference")]):
        axes[i].imshow(n, cmap="gray" if i<2 else "RdBu_r",
                       vmin=np.percentile(n,1), vmax=np.percentile(n,99))
        axes[i].set_title(title, fontsize=9)
        axes[i].axis("off")
    plt.suptitle("Noise Floor Analysis", fontsize=10, fontweight="bold")
    plt.tight_layout()

    return {
        "noise_std_a":    float(np.std(noise_a)),
        "noise_std_b":    float(np.std(noise_b)),
        "noise_mean_a":   float(np.mean(np.abs(noise_a))),
        "noise_mean_b":   float(np.mean(np.abs(noise_b))),
        "noise_diff_std": float(np.std(noise_a - noise_b)),
    }, fig

# 7. ENTROPY & TEXTURE
def analyze_texture(gray_a, gray_b):
    def tex(gray):
        g8 = (gray / 255 * 7).astype(np.uint8).clip(0,7)
        glcm = graycomatrix(g8, [1], [0, np.pi/4, np.pi/2], levels=8, symmetric=True, normed=True)
        return {
            "entropy":    float(shannon_entropy(gray)),
            "contrast":   float(graycoprops(glcm, "contrast").mean()),
            "homogeneity":float(graycoprops(glcm, "homogeneity").mean()),
            "energy":     float(graycoprops(glcm, "energy").mean()),
            "correlation":float(graycoprops(glcm, "correlation").mean()),
        }
    return {"yours": tex(gray_a), "theirs": tex(gray_b)}

# 8. DIFF HEATMAP
def make_diff_heatmap(arr_a, arr_b_matched):
    diff = np.abs(arr_a - arr_b_matched).mean(axis=2)
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    axes[0].imshow(arr_a.astype(np.uint8))
    axes[0].set_title("Yours (A)", fontsize=9); axes[0].axis("off")
    axes[1].imshow(arr_b_matched.astype(np.uint8))
    axes[1].set_title("Theirs (B)", fontsize=9); axes[1].axis("off")
    im = axes[2].imshow(diff, cmap="hot")
    axes[2].set_title("Pixel Difference Heatmap\nBright = bigger difference", fontsize=9)
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    plt.suptitle("Visual Difference Map", fontsize=10, fontweight="bold")
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════
# INTERPRET RESULTS & GIVE RECOMMENDATIONS
# ══════════════════════════════════════════════════════
def interpret(pixel, fft_r, dct_r, noise_r, tex_r):
    findings = []
    recs = []

    # Pixel diff
    mad = pixel["mean_abs_diff"]
    if mad < 2:
        findings.append(("PIXEL DIFF", "warn", f"Very small pixel difference ({mad:.2f}/255). "
            "Your processing barely changed the image — too gentle to disrupt SynthID."))
        recs.append("Increase perturbation strength. The changes need to be stronger at the pixel level.")
    elif mad > 15:
        findings.append(("PIXEL DIFF", "bad", f"Large pixel difference ({mad:.2f}/255). "
            "Your processing is too aggressive — causing visible quality loss."))
        recs.append("Reduce filter strength. You want pixel diff in the 3–8 range — enough to disrupt signal, not enough to see.")
    else:
        findings.append(("PIXEL DIFF", "good", f"Pixel difference looks reasonable ({mad:.2f}/255)."))

    # FFT mid-freq
    mfd = fft_r["mid_freq_diff"]
    if mfd < 0.3:
        findings.append(("FFT MID-FREQ", "bad", f"Almost no mid-frequency disruption ({mfd:.4f}). "
            "SynthID lives in mid frequencies — your tool is not targeting them."))
        recs.append("The working tool is disrupting mid-frequency bands. Add targeted FFT filtering at radial frequencies 0.1–0.6.")
    elif mfd > 1.5:
        findings.append(("FFT MID-FREQ", "warn", f"High mid-frequency disruption ({mfd:.4f}). "
            "Good disruption level but may be causing visible artifacts."))
    else:
        findings.append(("FFT MID-FREQ", "good", f"Good mid-frequency disruption ({mfd:.4f}). "
            "This is the target range for SynthID removal."))

    # DCT mid-freq
    mcf = dct_r["mid_freq_dct_diff"]
    lcf = dct_r["low_freq_dct_diff"]
    if mcf < lcf:
        findings.append(("DCT COEFFS", "bad", f"Low-freq DCT change ({lcf:.4f}) > mid-freq ({mcf:.4f}). "
            "You are changing the wrong coefficients — SynthID is in mid-range DCT, not low-freq."))
        recs.append("Your DCT perturbation is hitting the wrong bands. Target DCT indices [h//8:h//2, w//8:w//2] — avoid the DC and very-low coefficients.")
    else:
        findings.append(("DCT COEFFS", "good", f"Mid-freq DCT diff ({mcf:.4f}) correctly higher than low-freq ({lcf:.4f})."))

    # Noise floor
    na = noise_r["noise_std_a"]
    nb = noise_r["noise_std_b"]
    if nb > na * 1.5:
        findings.append(("NOISE FLOOR", "warn", f"Theirs has much higher noise ({nb:.3f} vs yours {na:.3f}). "
            "The working tool adds more noise — possibly using a stronger grain layer."))
        recs.append(f"Increase your noise/grain pass. Their noise floor std is {nb:.3f}, yours is {na:.3f}.")
    elif na > nb * 1.5:
        findings.append(("NOISE FLOOR", "bad", f"Your noise ({na:.3f}) is much higher than theirs ({nb:.3f}). "
            "You are adding too much noise causing graininess."))
        recs.append("Reduce your noise injection strength — you are over-noising the image.")
    else:
        findings.append(("NOISE FLOOR", "good", f"Noise floors are similar (yours {na:.3f}, theirs {nb:.3f})."))

    # Entropy
    ea = tex_r["yours"]["entropy"]
    eb = tex_r["theirs"]["entropy"]
    if eb > ea + 0.3:
        findings.append(("ENTROPY", "warn", f"Theirs has higher entropy ({eb:.3f} vs {ea:.3f}). "
            "Their output has more randomness/complexity — consistent with stronger perturbation."))
        recs.append("Their image has higher information entropy. This suggests a more thorough perturbation pass across the whole image.")
    else:
        findings.append(("ENTROPY", "good", f"Entropy similar (yours {ea:.3f}, theirs {eb:.3f})."))

    return findings, recs


# ══════════════════════════════════════════════════════
# BUILD PDF
# ══════════════════════════════════════════════════════
def build_pdf(path_a, path_b, output_pdf,
              file_r, pixel_r, fft_r, dct_r, noise_r, tex_r,
              findings, recs,
              fig_hist, fft_figs, fig_dct, fig_noise, fig_diff):

    doc = SimpleDocTemplate(output_pdf, pagesize=A4,
                            leftMargin=15*mm, rightMargin=15*mm,
                            topMargin=14*mm, bottomMargin=14*mm)
    story = []
    cw2 = (W - 36*mm) / 2

    # Title
    story.append(Paragraph("SynthID Forensic Analysis Report", ST["title"]))
    story.append(Paragraph(
        f"<b>A (Yours):</b> {os.path.basename(path_a)} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"<b>B (Theirs):</b> {os.path.basename(path_b)}", ST["sub"]))
    story.append(HRFlowable(width="100%", thickness=1.5,
                            color=colors.HexColor("#1a1a2e"), spaceAfter=8))

    # ── 1. File Info ──────────────────────────────────
    story.append(Paragraph("1. File &amp; Metadata", ST["h2"]))
    fa, fb = file_r["yours"], file_r["theirs"]
    rows = [
        ["Property", "Yours (A)", "Theirs (B)"],
        ["File size", f"{fa['size_kb']} KB", f"{fb['size_kb']} KB"],
        ["Dimensions", fa["dimensions"], fb["dimensions"]],
        ["Format", fa["format"], fb["format"]],
        ["EXIF tags present", str(len(fa["exif_keys"])), str(len(fb["exif_keys"]))],
    ]
    if fa["exif_keys"] or fb["exif_keys"]:
        rows.append(["EXIF keys", ", ".join(fa["exif_keys"]) or "none",
                     ", ".join(fb["exif_keys"]) or "none"])
    story.append(stat_table(rows, [50*mm, 65*mm, 65*mm]))
    story.append(Spacer(1, 3*mm))

    size_diff = abs(fa["size_kb"] - fb["size_kb"])
    if size_diff > 50:
        story.append(Paragraph(
            f"⚠ File size differs by {size_diff:.1f} KB. Larger file often means less "
            "aggressive JPEG compression was applied — which preserves more of the "
            "original pixel structure including the watermark.", ST["body"]))

    # ── 2. Pixel Stats ────────────────────────────────
    story.append(Paragraph("2. Pixel-Level Difference Statistics", ST["h2"]))
    rows2 = [
        ["Metric", "Value", "Meaning"],
        ["Mean absolute diff", f"{pixel_r['mean_abs_diff']:.4f} / 255",
         "Average per-pixel change. Target: 3–8"],
        ["Max diff", f"{pixel_r['max_diff']:.1f} / 255",
         "Largest single pixel change"],
        ["Std of diff", f"{pixel_r['std_diff']:.4f}",
         "Spread of changes across image"],
        ["PSNR", f"{pixel_r['psnr']:.2f} dB",
         "Quality loss. >35dB = invisible, <25dB = visible"],
        ["R channel mean diff", f"{pixel_r['mean_R']:.4f}", ""],
        ["G channel mean diff", f"{pixel_r['mean_G']:.4f}", ""],
        ["B channel mean diff", f"{pixel_r['mean_B']:.4f}", ""],
    ]
    story.append(stat_table(rows2, [55*mm, 55*mm, 60*mm]))
    story.append(Spacer(1, 3*mm))

    # ── 3. Histograms ─────────────────────────────────
    story.append(Paragraph("3. Color Channel Histograms", ST["h2"]))
    story.append(Paragraph(
        "Differences in histogram shape between A and B indicate colour processing "
        "differences. SynthID removal tools often subtly shift channel distributions "
        "as a side effect of their perturbations.", ST["body"]))
    story.append(fig_to_rl(fig_hist, 165, 62))

    # ── 4. FFT ────────────────────────────────────────
    story.append(Paragraph("4. Frequency Domain (FFT) Analysis", ST["h2"]))
    story.append(Paragraph(
        "This is the most important section. SynthID is embedded in the mid-frequency "
        "bands of the image. The FFT magnitude maps show where each image has energy. "
        "The difference map shows which tool disrupted more frequency content — and where. "
        "The radial profile shows the exact frequency bands each tool is targeting.", ST["body"]))
    story.append(fig_to_rl(fft_figs[0], 165, 55))
    story.append(Spacer(1, 2*mm))
    story.append(fig_to_rl(fft_figs[1], 165, 48))

    fft_rows = [
        ["FFT Metric", "Yours (A)", "Theirs (B)"],
        ["Mean magnitude", f"{fft_r['Yours']['mean_mag']:.4f}",
                           f"{fft_r['Theirs']['mean_mag']:.4f}"],
        ["Std magnitude",  f"{fft_r['Yours']['std_mag']:.4f}",
                           f"{fft_r['Theirs']['std_mag']:.4f}"],
        ["Mid-freq diff (A−B)", f"{fft_r['mid_freq_diff']:.4f}", "↑ higher = more disruption"],
        ["Overall FFT diff",    f"{fft_r['mean_diff']:.4f}",     "↑ higher = more disruption"],
    ]
    story.append(stat_table(fft_rows, [60*mm, 55*mm, 55*mm]))

    # ── 5. DCT ────────────────────────────────────────
    story.append(Paragraph("5. DCT Coefficient Analysis", ST["h2"]))
    story.append(Paragraph(
        "JPEG and SynthID both operate in the DCT domain. SynthID modifies mid-range "
        "DCT coefficients. If the working tool is disrupting different DCT bands than "
        "yours, that gap is likely why their removal works and yours doesn't.", ST["body"]))
    story.append(fig_to_rl(fig_dct, 165, 52))

    dct_rows = [
        ["DCT Band", "Diff magnitude", "Notes"],
        ["Low freq  (DC area)",  f"{dct_r['low_freq_dct_diff']:.6f}",  "Should be LOW — don't touch DC"],
        ["Mid freq  (target)",   f"{dct_r['mid_freq_dct_diff']:.6f}",  "Should be HIGH — this is SynthID's home"],
        ["High freq",            f"{dct_r['high_freq_dct_diff']:.6f}", "Less important for SynthID"],
        ["Overall mean",         f"{dct_r['mean_abs_dct_diff']:.6f}",  ""],
    ]
    story.append(stat_table(dct_rows, [55*mm, 55*mm, 60*mm]))

    # ── 6. Noise ──────────────────────────────────────
    story.append(Paragraph("6. Noise Floor Analysis", ST["h2"]))
    story.append(Paragraph(
        "The noise floor shows what each tool leaves behind after processing. "
        "SynthID tools typically add a calibrated noise layer — too little and the "
        "watermark survives, too much and the image looks grainy.", ST["body"]))
    story.append(fig_to_rl(fig_noise, 165, 52))

    noise_rows = [
        ["Noise Metric", "Yours (A)", "Theirs (B)"],
        ["Noise std dev",      f"{noise_r['noise_std_a']:.4f}",  f"{noise_r['noise_std_b']:.4f}"],
        ["Mean abs noise",     f"{noise_r['noise_mean_a']:.4f}", f"{noise_r['noise_mean_b']:.4f}"],
        ["Noise diff std",     f"{noise_r['noise_diff_std']:.4f}", "↑ = different noise patterns"],
    ]
    story.append(stat_table(noise_rows, [60*mm, 55*mm, 55*mm]))

    # ── 7. Texture & Entropy ──────────────────────────
    story.append(Paragraph("7. Texture &amp; Entropy Analysis", ST["h2"]))
    ta, tb = tex_r["yours"], tex_r["theirs"]
    tex_rows = [
        ["Metric", "Yours (A)", "Theirs (B)", "Notes"],
        ["Entropy",      f"{ta['entropy']:.4f}",      f"{tb['entropy']:.4f}",
         "Higher = more randomness"],
        ["Contrast",     f"{ta['contrast']:.4f}",     f"{tb['contrast']:.4f}",
         "Local intensity variation"],
        ["Homogeneity",  f"{ta['homogeneity']:.4f}",  f"{tb['homogeneity']:.4f}",
         "Smoothness of texture"],
        ["Energy",       f"{ta['energy']:.4f}",       f"{tb['energy']:.4f}",
         "Uniformity of texture"],
        ["Correlation",  f"{ta['correlation']:.4f}",  f"{tb['correlation']:.4f}",
         "Linear dependency of grey levels"],
    ]
    story.append(stat_table(tex_rows, [45*mm, 40*mm, 40*mm, 45*mm]))

    # ── 8. Heatmap ────────────────────────────────────
    story.append(Paragraph("8. Visual Difference Heatmap", ST["h2"]))
    story.append(Paragraph(
        "Bright areas in the heatmap show where the two images differ most. "
        "A good SynthID removal tool produces changes spread evenly across the "
        "entire image — not concentrated in corners or specific regions.", ST["body"]))
    story.append(fig_to_rl(fig_diff, 165, 58))

    # ── 9. Findings & Recommendations ─────────────────
    story.append(HRFlowable(width="100%", thickness=1.2,
                            color=colors.HexColor("#1a1a2e"), spaceAfter=6))
    story.append(Paragraph("9. Key Findings &amp; Recommendations", ST["h2"]))

    for (topic, level, text) in findings:
        icon = {"good":"✅","bad":"❌","warn":"⚠"}[level]
        story.append(Paragraph(f"{icon} <b>{topic}:</b> {text}", ST[level]))

    if recs:
        story.append(Spacer(1, 4*mm))
        story.append(Paragraph("What to change in your script:", ST["h2"]))
        for i, r in enumerate(recs, 1):
            story.append(Paragraph(f"{i}. {r}", ST["body"]))

    story.append(Spacer(1, 4*mm))
    story.append(HRFlowable(width="100%", thickness=0.8,
                            color=colors.HexColor("#cccccc"), spaceAfter=5))
    story.append(Paragraph(
        "This report is for technical research into digital watermarking systems.",
        ST["foot"]))

    doc.build(story)
    print(f"\n✅ PDF saved: {output_pdf}")


# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="SynthID Forensic Comparator — compare your output vs a working tool's output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python synthid_forensics.py --a my_output.jpg --b working_tool.jpg
  python synthid_forensics.py --a my_output.jpg --b working_tool.jpg --report analysis.pdf
        """
    )
    parser.add_argument("--a", required=True, help="YOUR processed image (still detected)")
    parser.add_argument("--b", required=True, help="Working tool's output (not detected)")
    parser.add_argument("--report", default="forensic_report.pdf", help="Output PDF path")
    args = parser.parse_args()

    print("\n" + "═"*54)
    print("  SynthID Forensic Comparator")
    print("═"*54)
    print(f"  A (yours) : {args.a}")
    print(f"  B (theirs): {args.b}")
    print("═"*54 + "\n")

    print("  Loading images...")
    img_a, arr_a, gray_a = load(args.a)
    img_b, arr_b, gray_b = load(args.b)

    print("  [1/8] File & metadata...")
    file_r = analyze_file(args.a, args.b)

    print("  [2/8] Pixel statistics...")
    pixel_r = analyze_pixels(arr_a, arr_b)
    arr_b_m = pixel_r["arr_b_matched"]

    # Resize gray_b if needed
    if gray_a.shape != gray_b.shape:
        h, w = gray_a.shape
        gray_b = np.array(Image.fromarray(gray_b.astype(np.uint8)).resize(
            (w,h), Image.LANCZOS), dtype=np.float32)

    print("  [3/8] Color histograms...")
    fig_hist = analyze_histogram(arr_a, arr_b_m)

    print("  [4/8] FFT frequency analysis...")
    fft_r, fft_figs = analyze_fft(gray_a, gray_b)

    print("  [5/8] DCT coefficient analysis...")
    dct_r, fig_dct = analyze_dct(gray_a, gray_b)

    print("  [6/8] Noise floor analysis...")
    noise_r, fig_noise = analyze_noise(gray_a, gray_b)

    print("  [7/8] Texture & entropy...")
    tex_r = analyze_texture(gray_a, gray_b)

    print("  [8/8] Diff heatmap...")
    fig_diff = make_diff_heatmap(arr_a, arr_b_m)

    print("\n  Interpreting results...")
    findings, recs = interpret(pixel_r, fft_r, dct_r, noise_r, tex_r)

    print("\n" + "─"*54)
    print("  FINDINGS SUMMARY")
    print("─"*54)
    icons = {"good":"✅","bad":"❌","warn":"⚠ "}
    for topic, level, text in findings:
        print(f"  {icons[level]} {topic}: {text}")

    if recs:
        print("\n  RECOMMENDATIONS:")
        for i, r in enumerate(recs, 1):
            print(f"  {i}. {r}")

    print("\n  Building PDF report...")
    build_pdf(args.a, args.b, args.report,
              file_r, pixel_r, fft_r, dct_r, noise_r, tex_r,
              findings, recs,
              fig_hist, fft_figs, fig_dct, fig_noise, fig_diff)

if __name__ == "__main__":
    main()
