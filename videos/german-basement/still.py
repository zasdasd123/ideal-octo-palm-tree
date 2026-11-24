#!/usr/bin/env python3
"""Full-resolution still of one shot frame: still.py <shot> <u 0..1> <spp> <out.png>"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import shots as SH  # noqa: E402
from pipeline.vox import voxlib as V  # noqa: E402

name, u, spp, out = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
wd = SH.Worlds()
s = next(x for x in SH.timeline() if x["name"] == name)
n = s["end"] - s["start"]
t = (s["start"] + np.arange(n)) / SH.FPS
key, f = SH.PROGRAMS[name](wd, t, np.arange(n) / max(n - 1, 1))
i = int(round(u * (n - 1)))
path, _ = wd.get(key)
t0 = time.time()
V.render(path, f[i:i + 1], out, W=1080, H=1920, spp=spp)
print(f"{name}[{i}] spp={spp}: {time.time() - t0:.2f}s")
