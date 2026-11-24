"""Scene look-dev stills: python -m pipeline.vox.lookdev <outdir> [scene ...]"""
import os
import sys
import time

import numpy as np

from pipeline.vox import scenes as S
from pipeline.vox import voxlib as V

out = sys.argv[1]
which = sys.argv[2:] or ["basement", "tower", "plains", "desert", "snow", "night", "nether", "end"]
os.makedirs(out, exist_ok=True)
pal = V.Palette()
W, H = 540, 960

for name in which:
    t0 = time.time()
    if name == "basement":
        w = S.basement(pal)
        path = f"{out}/{name}.bin"
        w.save(path)
        b = S.BASEMENT
        sx, sy, sz = b["stack"]
        cams = [((15.2, 5.6, 18.5), (sx, 5.6, sz), -1e9), ((12.6, 6.6, 13.0), (sx, 5.2, sz), 4.5),
                ((16.6, 7.1, 19.5), (sx, 5.4, sz), 6.2), ((14.4, 5.6, 21.0), (sx, 5.8, sz), 7.7)]
        for k, (pos, tgt, flood) in enumerate(cams):
            f = V.frames(1, "basement", ZEN=(0.30, 0.52, 0.98), HOR=(0.70, 0.84, 1.0), FOG0=150, FOG1=200)
            yaw, pitch = V.look_at(np.array(pos), np.array(tgt))
            V.set_cam(f, pos, yaw, pitch, fov=62, t=1.3)
            V.set_obj(f, 0, (sx, sy, sz), phase=k)
            V.set_obj(f, 1, (sx, sy + 1, sz), phase=k)
            f[:, V.P["FLOODY"]] = flood
            f[:, V.P["FLOODR"]:V.P["FLOODR"] + 4] = (b["x0"], b["x1"] + 1, b["z0"], b["z1"] + 1)
            f[:, V.P["WTINT"]:V.P["WTINT"] + 3] = (0.20, 0.42, 0.50)
            f[:, V.P["WDENS"]] = 0.16
            f[:, V.P["LAMP"]:V.P["LAMP"] + 3] = (b["lamp"][0] + 0.5, b["lamp"][1], b["lamp"][2] + 0.5)
            f[:, V.P["SPEC"]] = 0.6
            V.render(path, f, f"{out}/{name}_{k}.png", W=W, H=H, spp=4)
    elif name == "tower":
        w = S.tower(pal)
        path = f"{out}/{name}.bin"
        w.save(path)
        for k, (pos, tgt) in enumerate([((56, 24.0, 58), (64, 74, 118)), ((60, 34.0, 84), (64, 66, 118))]):
            f = V.frames(1, "day")
            yaw, pitch = V.look_at(np.array(pos, float), np.array(tgt, float))
            V.set_cam(f, pos, yaw, pitch, fov=62, t=1.0)
            V.render(path, f, f"{out}/{name}_{k}.png", W=W, H=H, spp=4)
    else:
        w, run = S.parkour(pal, name, seed=11 + len(name), length=12.0, fall_at=9.0 if name == "plains" else None)
        path = f"{out}/{name}.bin"
        w.save(path)
        times = np.array([2.0, 4.3, 6.1, 10.4] if name == "plains" else [2.0, 4.3, 6.1])
        pos, yaw, pitch = run.sample(times)
        f = V.frames(len(times), S.THEMES[name]["look"])
        V.set_cam(f, pos, yaw, pitch, fov=80, t=times)
        for k in range(len(times)):
            V.render(path, f[k:k + 1], f"{out}/{name}_{k}.png", W=W, H=H, spp=4)
    print(f"{name}: {time.time() - t0:.1f}s", file=sys.stderr)
