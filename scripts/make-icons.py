#!/usr/bin/env python3
"""Generate the Lumen PWA icon set (favicon, iOS apple-touch-icon, manifest icons)."""
import io
from PIL import Image, ImageDraw, ImageFilter

SIZES = [32, 180, 192, 512]          # favicon / apple-touch / android / full
MASKABLE = 512
OUT = "public/icons"

def radial(draw, cx, cy, r, color, alpha=255):
    """Soft radial glow via scaled stacked circles."""
    for i in range(r, 0, -1):
        a = int(alpha * (1 - i / r) ** 2)
        draw.ellipse([cx - i, cy - i, cx + i, cy + i], fill=color + (a,))

def make(size):
    img = Image.new("RGBA", (size, size), (10, 14, 28, 255))
    d = ImageDraw.Draw(img)
    # deep space gradient overlay
    glow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    radial(gd, int(size * 0.5), int(size * 0.42), int(size * 0.85), (34, 211, 238), 90)    # cyan
    radial(gd, int(size * 0.62), int(size * 0.68), int(size * 0.5), (139, 92, 246), 70)   # violet
    img = Image.alpha_composite(img, glow.filter(ImageFilter.GaussianBlur(size * 0.06)))
    d = ImageDraw.Draw(img)
    # aperture / lens ring
    cx, cy, R, r = size * 0.5, size * 0.5, size * 0.36, size * 0.14
    d.ellipse([cx - R, cy - R, cx + R, cy + R], outline=(226, 232, 240, 255), width=max(2, size // 32))
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(56, 189, 248, 255))
    d.ellipse([cx - r * 0.45, cy - r * 0.45, cx + r * 0.45, cy + r * 0.45], fill=(10, 14, 28, 255))
    # sparkle
    d.line([cx + R * 0.9, cy, cx + R * 1.25, cy], fill=(226, 232, 240, 255), width=max(1, size // 64))
    return img

import os
os.makedirs(OUT, exist_ok=True)
for s in SIZES:
    im = make(s)
    im.save(f"{OUT}/icon-{s}.png")
    if s == 512:
        # maskable: content in safe zone (80% circle), full-bleed bg
        im.save(f"{OUT}/icon-maskable-512.png")
print("icons written to", OUT, sorted(os.listdir(OUT)))