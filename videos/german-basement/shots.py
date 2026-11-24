#!/usr/bin/env python3
"""Background shots for "German's basement": camera programs cut to the voiceover.

    shots.py [--preview] [--only name,name] [--force]

Renders every shot with the in-repo voxel engine to build/shots/<name>.mp4 and writes
build/shots/timeline.json (frame ranges) for the compositor.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
from pipeline.vox import scenes as S  # noqa: E402
from pipeline.vox import voxlib as V  # noqa: E402

BUILD = os.path.join(HERE, "build")
FPS = 60
P = V.P
VO = json.load(open(os.path.join(HERE, "vo", "words.json"), encoding="utf-8"))
SENT = {}
for _w in VO["words"]:
    SENT.setdefault(_w["sent"], []).append(_w)


def W(s, k=0):
    return SENT[s][k]["s"]


def E(s, k=-1):
    return SENT[s][k]["e"]


CUT = 0.05
SHOTS = [
    ("tower", 0.0), ("reveal", W(2) - CUT), ("igor", W(3) - CUT), ("head", W(5) - CUT), ("slab", W(6) - CUT),
    ("argentina", W(6, 5) - CUT), ("orbit", W(8) - CUT), ("spin", W(10) - CUT), ("space", W(11) - CUT),
    ("tps", W(12) - CUT), ("flood_a", W(13) - CUT), ("flood_b", W(14) - CUT), ("under", W(15) - CUT),
    ("support", W(17) - CUT), ("ticket", W(18, 4) - CUT), ("igor2", W(19) - CUT), ("stars", W(20) - CUT),
    ("trial", W(22) - CUT), ("outro", W(24) - 0.10),
]
END = E(25) + 1.55


def timeline():
    fr = [int(round(t * FPS)) for _, t in SHOTS] + [int(round(END * FPS))]
    return [dict(name=n, start=fr[i], end=fr[i + 1], t0=fr[i] / FPS) for i, (n, _) in enumerate(SHOTS)]


def smooth(u):
    u = np.clip(u, 0, 1)
    return u * u * (3 - 2 * u)


def ease_out(u, p=2.4):
    return 1 - (1 - np.clip(u, 0, 1)) ** p


def lerp(a, b, u):
    return np.asarray(a, float)[None] * (1 - u[:, None]) + np.asarray(b, float)[None] * u[:, None]


def handheld(t, amp, seed=0.0):
    return np.stack([amp * np.sin(1.7 * t + seed) + amp * 0.5 * np.sin(4.1 * t + 2 * seed),
                     amp * 0.7 * np.sin(2.3 * t + 1.1 + seed), amp * 0.6 * np.sin(1.3 * t + 0.4 + seed)], 1)


class Worlds:
    """Lazily built scenes (saved under build/scenes)."""

    def __init__(self):
        self.pal = V.Palette()
        self.cache = {}
        os.makedirs(os.path.join(BUILD, "scenes"), exist_ok=True)

    def get(self, key):
        if key in self.cache:
            return self.cache[key]
        path = os.path.join(BUILD, "scenes", key + ".bin")
        run = None
        if key == "basement":
            w = S.basement(self.pal)
        elif key == "tower":
            w = S.tower(self.pal)
        else:
            seeds = dict(plains=17, desert=17, snow=15, night=16, nether=17, end=14)
            w, run = S.parkour(self.pal, key, seeds[key], length=24.0 if key == "plains" else 14.0,
                               fall_at=15.0 if key == "plains" else None)
            if key == "plains":      # the stack flies through this sky in the "space" shot
                w.add_object([[["washer"]]])
                w.add_object([[["server_lo"]], [["server_hi"]]])
        w.save(path)
        self.cache[key] = (path, run)
        return self.cache[key]


# ------------------------------------------------------------------ basement helpers

B = S.BASEMENT
STACK = np.array(B["stack"])


def basement_frames(n, t, flood=None):
    f = V.frames(n, "basement")
    f[:, P["TIME"]] = t
    f[:, P["FLOODR"]:P["FLOODR"] + 4] = (B["x0"], B["x1"] + 1, B["z0"], B["z1"] + 1)
    f[:, P["WTINT"]:P["WTINT"] + 3] = (0.20, 0.44, 0.52)
    f[:, P["WDENS"]] = 0.15
    f[:, P["LAMP"]:P["LAMP"] + 3] = (B["lamp"][0] + 0.5, B["lamp"][1], B["lamp"][2] + 0.5)
    f[:, P["SPEC"]] = 0.35
    if flood is not None:
        f[:, P["FLOODY"]] = flood
    return f


def set_stack(f, t, amp=None, drum=None, slide=None, turn=None, base=None):
    """Washer (object 0) + server (object 1). amp = shake amplitude in blocks."""
    n = len(t)
    amp = np.zeros(n) if amp is None else amp
    drum = t * 2.0 if drum is None else drum
    slide = np.zeros((n, 2)) if slide is None else slide
    turn = np.zeros(n) if turn is None else turn
    base = np.tile(STACK, (n, 1)) if base is None else base
    tau = 2 * np.pi
    for i in range(n):
        a, ti = amp[i], t[i]
        wp = base[i] + np.array([a * np.sin(tau * 11.3 * ti), 0, a * 0.7 * np.sin(tau * 13.7 * ti + 1.3)])
        Rw = V.rot_z(a * 0.9 * np.sin(tau * 9.1 * ti)) @ V.rot_x(a * 0.7 * np.sin(tau * 10.3 * ti + 0.7))
        V.set_obj(f[i], 0, wp, Rw, phase=drum[i])
        top = wp + Rw @ np.array([slide[i, 0], 1.0 + a * 0.55 * abs(np.sin(tau * 7.7 * ti)), slide[i, 1]])
        Rs = Rw @ V.rot_y(turn[i]) @ V.rot_z(a * 1.4 * np.sin(tau * 8.3 * ti + 2.1)) @ V.rot_x(a * 1.1 * np.sin(tau * 9.7 * ti))
        V.set_obj(f[i], 1, top, Rs, phase=ti * 6.0)


def aim(f, pos, target, fov, roll=0.0):
    yaw, pitch = V.look_at(pos, target)
    V.set_cam(f, pos, yaw, pitch, roll=roll, fov=fov)


# ------------------------------------------------------------------ shot programs
# each returns (world key, frames); t = absolute video time per frame, u = 0..1 inside the shot

def shot_tower(wd, t, u):
    f = V.frames(len(t), "day", TIME=0)
    f[:, P["TIME"]] = t * 4
    e = ease_out(u, 2.0)
    pos = lerp((54.5, 23.7, 50), (60.5, 37, 88), e) + handheld(t, 0.05)
    aim(f, pos, lerp((64, 84, 118), (64, 72, 118), e), 66 - 7 * e)
    return "tower", f


def shot_reveal(wd, t, u):
    f = basement_frames(len(t), t)
    e = ease_out(u, 3.0)
    pos = lerp((16.4, 7.2, 9.6), (15.25, 5.55, 19.2), e) + handheld(t, 0.03, 1.0)
    aim(f, pos, lerp((15.5, 4.6, 24.5), (15.5, 5.55, 24.5), e), 68 - 7 * e, roll=0.05 * (1 - e))
    set_stack(f, t)
    return "basement", f


def run_frames(wd, key, t, run_t, fov=80.0, roll=None):
    _, run = wd.get(key)
    f = V.frames(len(t), S.THEMES[key]["look"])
    pos, yaw, pitch = run.sample(run_t)
    V.set_cam(f, pos, yaw, pitch, roll=0.0 if roll is None else roll, fov=fov, t=t)
    return key, f


def shot_igor(wd, t, u):
    tt = t - t[0]
    lag0 = W(4, 0) - t[0]                 # "В игре": the run starts rubber-banding
    period, back, hold = 0.28, 0.19, 0.05
    k = np.floor(np.maximum(tt - lag0, 0) / period)
    into = np.maximum(tt - lag0, 0) - k * period
    lagged = lag0 + k * (period - back) + np.maximum(into - hold, 0)
    rt = np.where(tt < lag0, tt, lagged)
    return run_frames(wd, "plains", t, 1.6 + rt)


def shot_head(wd, t, u):
    return run_frames(wd, "desert", t, 3.2 + (t - t[0]))


def shot_slab(wd, t, u):
    f = basement_frames(len(t), t)
    e = u ** 2.2
    pos = lerp((15.5, 5.2, 20.4), (15.5, 7.5, 22.9), e) + handheld(t, 0.02, 2.0)
    V.set_cam(f, pos, 0.0, np.radians(8 + 78 * smooth(u * 1.25)), roll=0.25 * e, fov=64 + 20 * e)
    set_stack(f, t)
    return "basement", f


def shot_argentina(wd, t, u):
    return run_frames(wd, "snow", t, 2.4 + (t - t[0]) * 1.12, fov=84)


def shot_orbit(wd, t, u):
    f = basement_frames(len(t), t)
    phi = np.radians(-30 + 52 * smooth(u))
    r = 4.5 - 0.5 * u
    pos = np.stack([STACK[0] + r * np.sin(phi), 5.6 - 0.3 * u, STACK[2] - r * np.cos(phi)], 1) + handheld(t, 0.02, 3.0)
    aim(f, pos, np.array([15.5, 5.45, 24.5]), 60)
    set_stack(f, t)
    return "basement", f


def spin_curve(t):
    """Shake amplitude and drum phase: the spin cycle kicks in on "отжим"."""
    g = smooth((t - (W(10, 2) - 0.05)) / 1.9)
    amp = 0.006 + 0.125 * g ** 1.6
    rate = 2.0 + 34.0 * g                 # drum frames per second
    drum = np.concatenate([[0], np.cumsum(rate[1:] * np.diff(t))])
    return amp, drum, g


def shot_spin(wd, t, u):
    f = basement_frames(len(t), t)
    amp, drum, g = spin_curve(t)
    pos = lerp((14.2, 4.8, 21.7), (14.75, 4.62, 22.15), smooth(u)) + handheld(t, 0.012, 4.0)
    pos += (np.random.default_rng(5).normal(0, 1, (len(t), 3)) * (amp * 0.18)[:, None])
    aim(f, pos, np.array([15.5, 5.4, 24.5]), 62 - 4 * u, roll=0.02 * np.sin(31 * t) * g)
    slide = np.stack([0.16 * g, -0.20 * g], 1)
    set_stack(f, t, amp=amp, drum=drum, slide=slide, turn=0.42 * g)
    return "basement", f


def shot_space(wd, t, u):
    f = V.frames(len(t), "day")
    f[:, P["TIME"]] = t
    k = smooth(u * 1.15)
    f[:, P["ZEN"]:P["ZEN"] + 3] = lerp((0.30, 0.52, 0.98), (0.0, 0.0, 0.03), k)
    f[:, P["HOR"]:P["HOR"] + 3] = lerp((0.70, 0.84, 1.0), (0.03, 0.05, 0.17), k)
    f[:, P["FOG"]:P["FOG"] + 3] = f[:, P["HOR"]:P["HOR"] + 3]
    f[:, P["STARS"]] = k ** 2
    f[:, P["SUNSIZE"]] = 0
    y = 74 + 40 * u + 460 * u ** 2
    pos = np.stack([np.full_like(u, 64.0), y, np.full_like(u, 150.0)], 1)
    pitch = np.radians(58)
    V.set_cam(f, pos, 0.0, pitch, roll=0.10 * np.sin(5 * t), fov=70 + 8 * u)
    fwd = np.array([0, np.sin(pitch), np.cos(pitch)])
    up = np.array([0, np.cos(pitch), -np.sin(pitch)])
    dist = 5.2 + 5.5 * u ** 1.5
    amp, tau = 0.05, 2 * np.pi
    for i in range(len(t)):
        ti = t[i]
        R = V.rot_x(-pitch + 0.18 * np.sin(6 * ti)) @ V.rot_z(0.22 * np.sin(4.3 * ti)) @ V.rot_y(0.5 * np.sin(2.9 * ti) + 0.35)
        c = pos[i] + fwd * dist[i] + up * (0.5 - 1.4 * u[i]) + np.array([amp * np.sin(tau * 12 * ti), 0, 0])
        base = c - R @ np.array([0, 1.5, 0])
        V.set_obj(f[i], 0, base, R, phase=ti * 36)
        V.set_obj(f[i], 1, base + R @ np.array([0, 1.0, 0]), R, phase=ti * 6)
    return "plains", f


def shot_tps(wd, t, u):
    tt = t - t[0]
    stop = W(12, 1) - t[0]                # "в (пизду)": the server stops ticking altogether
    q = np.floor(np.minimum(tt, stop) * 7.5) / 7.5
    return run_frames(wd, "nether", t, 3.0 + q, fov=80)


def shot_flood_a(wd, t, u):
    f = basement_frames(len(t), t, flood=4.03 + 0.80 * u ** 1.6)
    pos = lerp((12.3, 6.6, 12.2), (12.9, 6.35, 14.6), smooth(u)) + handheld(t, 0.02, 5.0)
    aim(f, pos, np.array([15.6, 4.9, 24.5]), 62)
    set_stack(f, t)
    return "basement", f


def shot_flood_b(wd, t, u):
    f = basement_frames(len(t), t, flood=4.83 + 1.55 * smooth(u))
    pos = lerp((17.0, 7.25, 18.8), (16.6, 7.1, 20.3), smooth(u)) + handheld(t, 0.02, 6.0)
    aim(f, pos, np.array([15.5, 5.3, 24.5]), 62)
    set_stack(f, t)
    return "basement", f


def shot_under(wd, t, u):
    f = basement_frames(len(t), t, flood=7.2 + 0.6 * u)
    pos = lerp((14.1, 5.4, 20.4), (14.7, 5.75, 21.7), smooth(u)) + handheld(t, 0.05, 7.0)
    aim(f, pos, np.array([15.5, 5.9, 24.5]), 62, roll=0.06 * np.sin(1.9 * t))
    set_stack(f, t, amp=np.full(len(t), 0.004))
    return "basement", f


def shot_support(wd, t, u):
    return run_frames(wd, "night", t, 2.0 + (t - t[0]))


def shot_ticket(wd, t, u):
    return run_frames(wd, "end", t, 3.0 + (t - t[0]))


def shot_igor2(wd, t, u):
    return run_frames(wd, "desert", t, 7.4 + (t - t[0]))


def shot_stars(wd, t, u):
    return run_frames(wd, "plains", t, 8.6 + (t - t[0]))


def shot_trial(wd, t, u):
    _, run = wd.get("plains")
    tt = t - t[0]
    jump = W(23) - t[0]                   # "Дольше…": the jump that does not make it
    rt = run.fall_t - jump + tt
    air = np.clip(rt - run.fall_t - 0.45, 0, None)
    return run_frames(wd, "plains", t, rt, fov=80 + 10 * smooth(air / 0.8), roll=0.22 * np.sin(5.0 * air) * smooth(air / 0.5))


def shot_outro(wd, t, u):
    return run_frames(wd, "night", t, 8.2 + (t - t[0]) * 0.9)


PROGRAMS = dict(tower=shot_tower, reveal=shot_reveal, igor=shot_igor, head=shot_head, slab=shot_slab,
                argentina=shot_argentina, orbit=shot_orbit, spin=shot_spin, space=shot_space, tps=shot_tps,
                flood_a=shot_flood_a, flood_b=shot_flood_b, under=shot_under, support=shot_support,
                ticket=shot_ticket, igor2=shot_igor2, stars=shot_stars, trial=shot_trial, outro=shot_outro)


# slow, detailed shots get 4 rays per pixel; fast parkour hides aliasing and gets 2
SPP = dict(tower=4, reveal=4, slab=4, orbit=4, spin=4, space=4, flood_a=4, flood_b=4, under=4)
SEG = 48


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true", help="quarter-size, 1 spp, into build/preview")
    ap.add_argument("--only", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--budget", type=float, default=1e9, help="stop starting new segments after this many seconds")
    a = ap.parse_args()
    t_start = time.time()
    out = os.path.join(BUILD, "preview" if a.preview else "shots")
    os.makedirs(out, exist_ok=True)
    tl = timeline()
    json.dump(dict(fps=FPS, frames=int(round(END * FPS)), shots=tl), open(os.path.join(out, "timeline.json"), "w"), indent=1)
    wd = Worlds()
    only = set(a.only.split(",")) - {""}
    for s in tl:
        if only and s["name"] not in only:
            continue
        dst = os.path.join(out, s["name"] + ".mp4")
        if os.path.exists(dst) and not a.force:
            continue
        n = s["end"] - s["start"]
        t = (s["start"] + np.arange(n)) / FPS
        key, f = PROGRAMS[s["name"]](wd, t, np.arange(n) / max(n - 1, 1))
        path, _ = wd.get(key)
        # resumable: fixed-size segments, concatenated once they all exist
        parts = []
        for k, a0 in enumerate(range(0, n, SEG)):
            part = os.path.join(out, f"{s['name']}.p{k:03d}.mp4")
            parts.append(part)
            if os.path.exists(part) and not a.force:
                continue
            if time.time() - t_start > a.budget:
                print(f"PARTIAL: stopped before {s['name']} segment {k}", flush=True)
                return
            tmp = part[:-4] + ".tmp.mp4"
            if a.preview:
                V.render(path, f[a0:a0 + SEG], tmp, W=270, H=480, spp=1, crf=18)
            else:
                V.render(path, f[a0:a0 + SEG], tmp, W=1080, H=1920, spp=SPP.get(s["name"], 2), crf=10)
            os.replace(tmp, part)
        lst = dst[:-4] + ".txt"
        with open(lst, "w") as fh:
            fh.writelines(f"file '{p}'\n" for p in parts)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", dst], check=True)
        for p in parts + [lst]:
            os.remove(p)
        print(f"{s['name']}: {n} frames [{s['t0']:.2f}s] -> {dst} ({time.time() - t_start:.0f}s)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
