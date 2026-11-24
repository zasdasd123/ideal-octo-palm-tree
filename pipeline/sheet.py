#!/usr/bin/env python3
"""Contact sheet of clips: one row per clip, N evenly spaced frames. sheet.py out.png cols clip1.mp4 ..."""
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

out, cols, clips = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
rows = []
for c in clips:
    info = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                           "stream=width,height,nb_read_packets", "-of", "csv=p=0", c], capture_output=True, text=True).stdout.split(",")
    w, h, n = int(info[0]), int(info[1]), int(info[2])
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", c, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    idx = np.linspace(0, len(fr) - 1, cols).round().astype(int)
    th = 300
    tw = int(w * th / h)
    row = Image.new("RGB", (tw * cols, th))
    for k, i in enumerate(idx):
        row.paste(Image.fromarray(fr[i]).resize((tw, th), Image.BILINEAR), (k * tw, 0))
    ImageDraw.Draw(row).text((4, 2), f"{c.split('/')[-1]} ({n}f)", fill=(255, 255, 0))
    rows.append(row)
sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows)))
y = 0
for r in rows:
    sheet.paste(r, (0, y))
    y += r.height
sheet.save(out)
print(out, sheet.size)
