#!/usr/bin/env python3
"""Sound design + mix for "German's basement": original synthesised music and SFX under the voiceover.

    audio.py   ->  build/audio/mix.wav (+ stems music.wav, sfx.wav)
"""
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
import edit as ED  # noqa: E402
from pipeline import synth as S  # noqa: E402

SR = S.SR
C, CUT, W, E, END = ED.C, ED.CUT, ED.W, ED.E, ED.END
OUT = os.path.join(HERE, "build", "audio")
MUSIC_GAIN, SFX_GAIN = 0.305, 0.35      # bed sits ~10 LU under the -16 LUFS voice, effects a little above the bed


def stereo(fn, x, *a, **k):
    return np.stack([fn(x[:, 0], *a, **k), fn(x[:, 1], *a, **k)], 1)


def edges(x, a=0.003, r=0.004):
    x = x.copy()
    na, nr = int(a * SR), int(r * SR)
    x[:na] *= np.linspace(0, 1, na)[:, None]
    x[-nr:] *= np.linspace(1, 0, nr)[:, None]
    return x


def music():
    tr = S.Track(END)

    # 0.0  pompous "corporate" pad for the elite data centre, tape-stopped at the basement cut
    intro = S.Track(3.4)
    chord = [S.NOTE[n] for n in ("F3", "A3", "C4", "E4", "G4")]
    intro.add(0.0, S.pad(chord, 3.2, attack=0.10, bright=2600), 0.55)
    intro.add(0.0, S.pad([S.NOTE["F2"]], 3.2, attack=0.05, bright=500), 0.5)
    for k, n in enumerate(("C5", "E5", "G5", "C6", "E6", "G6", "C7", "G6", "E6", "C6", "G5", "E5")):
        intro.add(0.12 + k * 0.115 * 2, S.bell(S.NOTE[n], 0.7), 0.16, pan=(-0.5 if k % 2 else 0.5))
    stop = CUT["reveal"] - 0.24
    tr.buf[:len(intro.buf)] += stereo(S.tape_stop, intro.buf, stop, 0.27) * 0.9

    def section(t0, t1, variant, gain=1.0, fx=None, seed=0):
        b = S.beat(t1 - t0, variant=variant, seed=seed)
        if fx:
            b = fx(b)
        i = int(t0 * SR)
        b = edges(b)
        tr.buf[i:i + len(b)] += b * gain

    # 5.4  Igor: main loop; it rubber-bands with the gameplay and dies for the freeze-frame
    t0 = CUT["igor"]

    def lagfx(b):
        b = stereo(S.stutter, b, C["stutter"] - t0, 60 / 130 / 4, int((C["freeze1"] - C["stutter"]) / (60 / 130 / 4)) + 1)
        b[int((C["freeze1"] - t0) * SR):] = 0
        return b
    section(t0, CUT["head"], "main", fx=lagfx)

    # 9.7  "включи голову" -> concrete slab (muffled) -> Argentina
    t0 = CUT["head"]

    def slabfx(b):
        t = np.arange(len(b)) / SR + t0
        fc = np.full(len(b), 16000.0)
        m = (t > CUT["slab"]) & (t < CUT["argentina"])
        fc[m] = 520
        k = int(0.03 * SR)
        fc = np.convolve(fc, np.ones(k) / k, mode="same")
        return stereo(S.lowpass_sweep, b, fc)
    section(t0, CUT["orbit"], "main", fx=slabfx, seed=1)

    # 16.6  washing machine: sparse groove + snare roll into the turbo boost
    t0 = CUT["orbit"]
    section(t0, CUT["space"], "sparse", gain=0.95, seed=2)
    sn = S.snare()
    t, k = C["turbo"] - 1.25, 0
    while t < CUT["space"] - 0.03:
        u = (t - (C["turbo"] - 1.25)) / (CUT["space"] - C["turbo"] + 1.25)
        tr.add(t, sn, 0.22 + 0.42 * u, pan=0.1)
        t += 0.23 * (1 - u) + 0.045
        k += 1

    # 22.6  "в космос": full loop, then the tape runs out with the TPS
    t0 = CUT["space"]
    section(t0, C["freeze2"] + 0.1, "alt", gain=1.05, seed=3,
            fx=lambda b: stereo(S.tape_stop, b, CUT["tps"] - t0 + 0.05, C["freeze2"] - CUT["tps"] - 0.08))

    # 25.8  flood: half-time, sinking under water
    t0 = CUT["flood_a"]

    def floodfx(b):
        t = np.arange(len(b)) / SR + t0
        fc = np.interp(t, [t0, CUT["flood_b"], CUT["under"] - 0.05, CUT["under"] + 0.25, CUT["support"]],
                       [6500, 2600, 1300, 360, 300])
        return stereo(S.lowpass_sweep, b, fc)
    section(t0, CUT["support"], "half", gain=1.0, fx=floodfx, seed=4)

    # 33.3  support tickets: loop back, bone dry; stops dead for Igor's ticket
    section(CUT["support"], C["typed0"] - 0.05, "alt", seed=5)

    # 41.5  five stars -> offer -> the fall; cut to dead silence at the brand reveal
    section(C["stars"] - 0.10, C["outro"], "main", seed=6)
    return tr.buf[:int(END * SR)] * 0.5


def sfx():
    tr = S.Track(END)
    a = tr.add
    # a low thud under every swear word (the TPS line has its own hit)
    for ts, sent in ED.SWEAR_T:
        if sent != 12:
            a(ts, S.impact(105, 42, 0.28, 0.6), 0.34)
    # basement reveal
    a(CUT["reveal"] - 0.02, S.glitch(0.22, 3), 0.30)
    a(CUT["reveal"], S.impact(90, 34, 0.9, 0.5), 0.55)
    a(CUT["reveal"] + 0.55, S.drip(1500), 0.22, pan=-0.4)
    a(CUT["reveal"] + 1.35, S.drip(1250), 0.18, pan=0.5)
    a(C["label_basement"] + 0.02, S.impact(120, 45, 0.45), 0.60)
    amb = S.filt(S.noise((CUT["igor"] - CUT["reveal"]) * SR, 31), "lowpass", 140, 4) * 6
    a(CUT["reveal"], S.fade(amb, 0.2, 0.3), 0.18)
    # Igor / lag
    a(CUT["igor"] - 0.10, S.whoosh(0.24, 500, 6000, 4), 0.26)
    a(C["freeze1"], S.buzz(0.36), 0.42)
    a(C["freeze1"] - 0.01, S.impact(100, 40, 0.35), 0.45)
    a(CUT["head"] - 0.02, S.impact(130, 48, 0.4), 0.50)
    a(CUT["head"] - 0.12, S.whoosh(0.22, 800, 7000, 5), 0.22)
    # slab + Argentina
    a(CUT["slab"] - 0.08, S.whoosh(0.2, 2000, 300, 6), 0.24)
    a(CUT["slab"], S.impact(80, 34, 0.7, 0.3), 0.40)
    a(CUT["argentina"] - 0.34, S.riser(0.34, 7), 0.5)
    a(CUT["argentina"] - 0.01, S.impact(140, 50, 0.3), 0.42)
    a(W(6, 6) + 0.02, S.tango(), 0.85, pan=0.1)
    a(C["fast"] - 0.01, S.impact(120, 44, 0.35), 0.42)
    # washing machine
    a(CUT["orbit"] - 0.1, S.whoosh(0.22, 3000, 400, 8), 0.22)
    a(CUT["spin"] - 0.1, S.whoosh(0.2, 600, 5000, 9), 0.2)
    spin_len = CUT["space"] - C["spin"] + 0.05
    a(C["spin"] - 0.05, S.fade(S.washer(spin_len), 0.05, 0.015), 0.50)
    a(C["turbo"] - 0.01, S.impact(150, 52, 0.4), 0.50)
    # space
    a(CUT["space"] - 0.02, S.impact(160, 40, 0.8), 0.55)
    a(CUT["space"], S.whoosh(1.35, 250, 9000, 10), 0.34)
    zoom = np.sin(S.sweep(180, 2400, 1.3, 1.5))
    a(CUT["space"] + 0.1, zoom * np.linspace(0.3, 0, len(zoom)) ** 0.6, 0.20)
    # TPS dies
    a(CUT["tps"], S.glitch(0.16, 11), 0.26)
    a(CUT["tps"] + 0.05, S.powerdown(C["freeze2"] - CUT["tps"] - 0.05), 0.48)
    a(C["freeze2"] + 0.12, S.impact(70, 28, 0.9, 0.2), 0.62)
    a(C["freeze2"] + 0.12, S.buzz(0.22), 0.18)
    # flood
    a(CUT["flood_a"] + 0.05, S.drip(1400), 0.30, pan=-0.3)
    a(CUT["flood_a"] + 0.55, S.drip(1150), 0.28, pan=0.4)
    a(CUT["flood_a"] + 0.95, S.drip(1700), 0.24, pan=0.0)
    flow = S.water_flow(CUT["under"] - CUT["flood_a"])
    ramp = np.linspace(0.15, 1.0, len(flow)) ** 1.5
    a(CUT["flood_a"], S.fade(flow * ramp, 0.3, 0.05), 0.34)
    a(CUT["flood_a"] + 0.8, S.bubbles(CUT["under"] - CUT["flood_a"] - 0.8, 9, 21), 0.3)
    a(C["cool"] - 0.02, S.impact(120, 42, 0.5), 0.5)
    a(CUT["under"] - 0.02, S.impact(60, 26, 1.2, 0.1), 0.5)
    deep = S.filt(S.noise((CUT["support"] - CUT["under"]) * SR, 22), "lowpass", 220, 4) * 5
    a(CUT["under"], S.fade(deep, 0.05, 0.02), 0.18)
    a(CUT["under"] + 0.05, S.filt(S.bubbles(CUT["support"] - CUT["under"] - 0.1, 16, 23, 180, 700), "lowpass", 1400), 0.5)
    # tickets
    a(CUT["support"] - 0.04, S.whoosh(0.3, 300, 8000, 12), 0.36)
    a(CUT["support"], S.impact(140, 46, 0.45), 0.5)
    a(CUT["ticket"] - 0.1, S.whoosh(0.2, 900, 6000, 13), 0.18)
    for key in ("closed", "positive"):
        a(C[key], S.impact(170, 55, 0.3, 1.6), 0.62)
    a(CUT["igor2"] - 0.1, S.whoosh(0.2, 700, 5000, 14), 0.18)
    n_keys = 13
    for k in range(n_keys):                                   # typing
        tk = C["typed0"] + (C["typed1"] - C["typed0"]) * k / n_keys
        a(tk, S.filt(S.noise(500, 40 + k), "bandpass", (1800, 5200)) * np.exp(-np.arange(500) / 90), 0.16, pan=0.2)
    a(E(19) + 0.06, S.cowbell(S.NOTE["F5"], 0.4), 0.42)
    a(C["noq"] + 0.02, S.buzz(0.16), 0.2)
    for k, n in enumerate(("C6", "D6", "E6", "G6", "C7")):
        a(C["stars"] + k * 0.11, S.bell(S.NOTE[n], 0.5), 0.26, pan=-0.4 + 0.2 * k)
    # offer + fall
    a(CUT["trial"] - 0.1, S.whoosh(0.22, 600, 6000, 15), 0.2)
    a(C["free"], S.bell(S.NOTE["C6"], 0.9), 0.24)
    a(C["free"] + 0.07, S.bell(S.NOTE["G6"], 0.9), 0.2)
    a(W(22, 5), S.impact(130, 46, 0.35), 0.4)
    rl = C["outro"] - W(23)
    a(W(23), S.riser(rl, 16), 0.62)
    wind = S.whoosh(rl - 0.2, 200, 3500, 17)
    a(W(23) + 0.2, wind * np.linspace(0.2, 1, len(wind)) ** 2, 0.55)
    a(C["dead"], S.impact(90, 30, 0.3, 0.4), 0.5)
    # outro: music is gone; only the hits that sell the logo
    a(C["outro"], S.impact(72, 30, 0.38, 0.5), 0.78)       # one dry thump on the cut, then the voice is alone
    a(C["badge"] + 0.02, S.shimmer(0.6), 0.20)
    a(C["badge"], S.impact(150, 48, 0.4, 1.4), 0.55)
    a(END - 0.34, S.powerdown(0.30, 420, 40), 0.40)       # the picture dies with a CRT whine
    return tr.buf[:int(END * SR)]


def lufs(path):
    out = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-af", "ebur128=peak=true", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    return float(re.findall(r"I:\s+(-?[0-9.]+) LUFS", out)[-1]), float(re.findall(r"Peak:\s+(-?[0-9.]+) dBFS", out)[-1])


def write(path, x):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f64le", "-ar", str(SR), "-ac", str(x.shape[1]), "-i", "-",
                    "-c:a", "pcm_s24le", path], input=np.ascontiguousarray(x).tobytes(), check=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    n = int(END * SR)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", os.path.join(HERE, "vo", "vo.wav"), "-ac", "1",
                          "-ar", str(SR), "-f", "f64le", "-"], capture_output=True, check=True).stdout
    vo = np.zeros(n)
    v = np.frombuffer(raw, np.float64)
    vo[:min(n, len(v))] = v[:n]
    mus, fx = music(), sfx()
    # dead silence under the brand line: nothing musical survives the outro cut
    mus[int(C["outro"] * SR):] = 0
    mus *= MUSIC_GAIN
    fx *= SFX_GAIN
    # duck the music under the voice (about 4 dB)
    env = np.sqrt(np.convolve(vo ** 2, np.ones(2400) / 2400, mode="same"))
    k = int(0.18 * SR)
    env = np.convolve(env, np.ones(k) / k, mode="same")
    duck = 1.0 - 0.36 * np.clip(env / (np.percentile(env, 90) + 1e-9), 0, 1)
    mus = mus * duck[:, None]
    mix = np.stack([vo, vo], 1) + mus + fx
    mix = np.tanh(mix * 0.9) / 0.9
    write(os.path.join(OUT, "music.wav"), mus)
    write(os.path.join(OUT, "sfx.wav"), fx)
    pre = os.path.join(OUT, "premaster.wav")
    write(pre, mix)
    i, _ = lufs(pre)
    final = os.path.join(OUT, "mix.wav")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", pre, "-af",
                    f"volume={-14.0 - i:.2f}dB,alimiter=limit=0.89:attack=2:release=50:level=false",
                    "-ar", str(SR), "-c:a", "pcm_s24le", final], check=True)
    for name in ("music", "sfx", "mix"):
        li, pk = lufs(os.path.join(OUT, name + ".wav"))
        print(f"{name}: {li:.1f} LUFS, peak {pk:.1f} dBFS")
    li, _ = lufs(os.path.join(HERE, "vo", "vo.wav"))
    print(f"vo: {li:.1f} LUFS; duration {END:.2f}s")
    # per-second level table (dBFS RMS): voice / music / sfx
    def db(x):
        return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-6)
    rows = []
    for s in range(int(END) + 1):
        a, b = s * SR, min((s + 1) * SR, n)
        rows.append(f"{s:02d}:{db(vo[a:b]):4.0f}/{db(mus[a:b]):4.0f}/{db(fx[a:b]):4.0f}")
    for i in range(0, len(rows), 9):
        print("  " + "  ".join(rows[i:i + 9]))


if __name__ == "__main__":
    main()
