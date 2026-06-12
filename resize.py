"""
Resize image before forensic comparison
Usage: python resize.py input.jpg 1440
"""
import sys
from PIL import Image
from pathlib import Path

path = Path(sys.argv[1])
size = int(sys.argv[2])

img = Image.open(path).convert("RGB")
img = img.resize((size, size), Image.LANCZOS)
out = path.with_stem(path.stem + f"_{size}")
img.save(str(out), format="JPEG", quality=92)
print(f"✅ Saved: {out}  ({size}x{size})")
