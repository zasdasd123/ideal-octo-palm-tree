"""Tiny synth toolkit: drums, 808, cowbell, pads and sound effects, all generated from code."""
import numpy as np
from scipy import signal

SR = 48000
TAU = 2 * np.pi


def t_(dur):
    return np.arange(int(round(dur * SR))) / SR


def noise(n, seed=0):
    return np.random.default_rng(seed).standard_normal(int(round(n))).astype(np.float64)


def put(out, i, seg, gain=1.0):
    n = min(len(seg), len(out) - i)
    if n > 0:
        out[i:i + n] += seg[:n] * gain


def filt(x, kind, freq, order=2):
    sos = signal.butter(order, np.asarray(freq) / (SR / 2), btype=kind, output="sos")
    return signal.sosfilt(sos, x)


def fade(x, a=0.004, r=0.01):
    x = x.copy()
    na, nr = min(int(a * SR), len(x) // 2), min(int(r * SR), len(x) // 2)
    if na:
        x[:na] *= np.linspace(0, 1, na)
    if nr:
        x[-nr:] *= np.linspace(1, 0, nr)
    return x


def sweep(f0, f1, dur, curve=1.0):
    n = int(round(dur * SR))
    u = (np.arange(n) / max(n - 1, 1)) ** curve
    f = f0 * (f1 / f0) ** u
    return np.cumsum(f) * TAU / SR


def kick(dur=0.30):
    ph = sweep(165, 46, dur, 0.35)
    x = np.sin(ph) * np.exp(-t_(dur) / 0.11)
    x[:96] += noise(96, 1) * np.linspace(0.5, 0, 96)
    return np.tanh(x * 1.6)


def snare(dur=0.22, seed=2):
    t = t_(dur)
    n = filt(noise(len(t), seed), "bandpass", (900, 7000)) * np.exp(-t / 0.055)
    tone = np.sin(TAU * 190 * t) * np.exp(-t / 0.04)
    clap = np.zeros(len(t))
    for k, d in enumerate((0.0, 0.011, 0.023)):
        i = int(d * SR)
        seg = filt(noise(1400, seed + k + 5), "bandpass", (1200, 5000)) * np.exp(-np.arange(1400) / (0.006 * SR))
        clap[i:i + 1400] += seg
    return np.tanh((n * 0.9 + tone * 0.5 + clap * 0.7) * 1.3)


def hat(dur=0.05, seed=3, open_=False):
    t = t_(0.22 if open_ else dur)
    x = filt(noise(len(t), seed), "highpass", 7500, 4) * np.exp(-t / (0.07 if open_ else 0.014))
    return x * 0.6


def cowbell(freq, dur=0.32):
    t = t_(dur)
    x = signal.square(TAU * freq * t) + 0.8 * signal.square(TAU * freq * 1.481 * t)
    x = filt(x, "bandpass", (freq * 0.9, min(freq * 4.2, 11000)))
    env = 0.75 * np.exp(-t / 0.018) + 0.45 * np.exp(-t / 0.12)
    return np.tanh(x * env * 1.4) * 0.5


def bass808(freq, dur):
    t = t_(dur + 0.08)
    f = freq * (1 + 1.2 * np.exp(-t / 0.012))
    ph = np.cumsum(f) * TAU / SR
    env = np.minimum(1, t / 0.004) * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.03)) * (0.8 + 0.2 * np.exp(-t / 0.25))
    x = np.tanh(np.sin(ph) * 2.6) * env
    return x * 0.8 + filt(np.tanh(np.sin(ph) * 6) * env, "highpass", 160) * 0.22   # upper harmonics for small speakers


def pad(freqs, dur, attack=0.12, bright=1400):
    t = t_(dur)
    x = np.zeros(len(t))
    for k, f in enumerate(freqs):
        for det in (-0.004, 0.0, 0.005):
            x += signal.sawtooth(TAU * f * (1 + det) * t + k) * 0.33
    x = filt(x, "lowpass", bright, 2) / len(freqs)
    return x * np.minimum(1, t / attack)


def bell(freq, dur=0.6, bright=1.0):
    t = t_(dur)
    x = np.sin(TAU * freq * t + bright * 1.2 * np.sin(TAU * freq * 2.76 * t) * np.exp(-t / 0.08))
    x += 0.35 * np.sin(TAU * freq * 2.0 * t) * np.exp(-t / 0.10)
    return x * np.exp(-t / (dur * 0.28)) * np.minimum(1, t / 0.002)


def whoosh(dur=0.28, f0=400, f1=5000, seed=7):
    n = noise(dur * SR, seed)
    u = np.linspace(0, 1, len(n))
    out = np.zeros(len(n))
    blk = 256
    zi = np.zeros((2, 2))
    for i in range(0, len(n), blk):
        fc = f0 * (f1 / f0) ** u[i]
        sos = signal.butter(2, (fc * 0.6 / (SR / 2), min(fc * 1.7, SR * 0.45) / (SR / 2)), btype="bandpass", output="sos")
        out[i:i + blk], zi = signal.sosfilt(sos, n[i:i + blk], zi=zi)
    return out * np.sin(np.pi * u) ** 1.5


def impact(f0=110, f1=38, dur=0.5, click=1.0, seed=8):
    t = t_(dur)
    x = np.sin(sweep(f0, f1, dur, 0.4)) * np.exp(-t / (dur * 0.33))
    c = filt(noise(len(t), seed), "bandpass", (1500, 6000)) * np.exp(-t / 0.012) * click
    return np.tanh((x * 1.5 + c * 0.6) * 1.4)


def glitch(dur=0.2, seed=9):
    r = np.random.default_rng(seed)
    out = np.zeros(int(dur * SR))
    i = 0
    while i < len(out):
        ln = int(r.integers(300, 1900))
        f = float(r.choice([110, 220, 440, 880, 1760, 3520]))
        seg = signal.square(TAU * f * np.arange(ln) / SR) if r.random() < 0.6 else np.sign(noise(ln, int(r.integers(1e6))))
        hold = int(r.integers(2, 12))
        seg = np.repeat(seg[::hold], hold)[:ln]
        put(out, i, seg, r.uniform(0.3, 1.0) * (r.random() < 0.85))
        i += ln
    return fade(out, 0.001, 0.004) * 0.5


def buzz(dur=0.38):
    t = t_(dur)
    x = signal.square(TAU * 147 * t) + signal.square(TAU * 156 * t)
    return fade(filt(x, "lowpass", 2600), 0.003, 0.03) * 0.35


def drip(f=1500, seed=10):
    t = t_(0.5)
    x = np.sin(sweep(f, f * 0.45, 0.5, 0.25)) * np.exp(-t / 0.035)
    d = int(0.13 * SR)
    x[d:] += x[:-d] * 0.35
    x[2 * d:] += x[:-2 * d] * 0.15
    return x * 0.5


def bubbles(dur, rate=14, seed=11, f_lo=250, f_hi=1100):
    r = np.random.default_rng(seed)
    out = np.zeros(int(dur * SR))
    for _ in range(int(dur * rate)):
        i = int(r.uniform(0, max(dur - 0.12, 0.01)) * SR)
        f = r.uniform(f_lo, f_hi)
        ln = int(r.uniform(0.03, 0.09) * SR)
        seg = np.sin(sweep(f, f * r.uniform(1.4, 2.2), ln / SR))
        seg = seg * np.hanning(len(seg)) * r.uniform(0.3, 1.0)
        put(out, i, seg)
    return out * 0.35


def water_flow(dur, seed=12):
    t = t_(dur)
    n = filt(noise(len(t), seed), "bandpass", (300, 3200))
    lfo = 0.6 + 0.4 * np.sin(TAU * 0.9 * t + 1.0)
    return n * lfo * 0.25


def washer(dur, seed=13):
    """Spin cycle winding up: motor whine, drum rumble and a rattling cabinet."""
    t = t_(dur)
    g = np.clip(t / dur, 0, 1) ** 1.4
    whine = signal.sawtooth(sweep(70, 520, dur, 1.3)) * 0.5 + 0.3 * np.sin(sweep(140, 1040, dur, 1.3))
    whine = filt(whine, "lowpass", 2600)
    rot = np.cumsum(2 + 16 * g) * TAU / SR
    rumble = filt(noise(len(t), seed), "lowpass", 180, 4) * (0.55 + 0.45 * np.sin(rot)) * 3.0
    r = np.random.default_rng(seed)
    rattle = np.zeros(len(t))
    i = 0
    while i < len(t) - 2000:
        f = r.uniform(1800, 5200)
        seg = np.sin(TAU * f * np.arange(1500) / SR) * np.exp(-np.arange(1500) / (0.004 * SR))
        put(rattle, i, seg, r.uniform(0.3, 1.0) * g[i])
        i += int(SR / (5 + 26 * g[i]) * r.uniform(0.7, 1.3))
    return (whine * 0.5 + rumble * 0.8 + rattle * 0.45) * (0.25 + 0.75 * g)


def riser(dur, seed=14):
    t = t_(dur)
    u = t / dur
    n = filt(noise(len(t), seed), "highpass", 1200) * u ** 2 * 0.35
    tone = signal.sawtooth(sweep(110, 880, dur, 1.6)) * u ** 2 * 0.22
    return filt(n + tone, "lowpass", 9000)


def powerdown(dur=0.9, f0=520, f1=38):
    t = t_(dur)
    x = signal.sawtooth(sweep(f0, f1, dur, 0.7)) * (1 - 0.6 * t / dur)
    return fade(filt(x, "lowpass", 2200), 0.004, 0.08) * 0.4


def shimmer(dur=1.1, seed=15):
    t = t_(dur)
    x = filt(noise(len(t), seed), "bandpass", (5200, 11500), 3) * np.exp(-t / 0.28)
    for f in (2093, 3136, 4186):
        x += np.sin(TAU * f * t) * np.exp(-t / 0.35) * 0.12
    return x * 0.3


def tango(root=146.83):
    """Two bars of the tango 'marcato' on a reedy squeezebox: the Argentina gag."""
    chord = (1.0, 1.189, 1.498, 2.0)                      # minor triad + octave
    hits = [(0.00, 0.11), (0.21, 0.07), (0.32, 0.11), (0.50, 0.16)]
    out = np.zeros(int(0.9 * SR))
    for st, ln in hits:
        t = t_(ln + 0.05)
        x = np.zeros(len(t))
        for k, m in enumerate(chord):
            f = root * 2 * m
            x += signal.square(TAU * f * t, duty=0.32) * 0.5 + signal.sawtooth(TAU * f * 1.006 * t + k) * 0.5
        x = filt(x, "bandpass", (350, 3600)) * np.minimum(1, t / 0.006) * np.where(t < ln, 1, np.exp(-(t - ln) / 0.012))
        put(out, int(st * SR), x, 0.12)
    return out


def lowpass_sweep(x, fc):
    """Two cascaded one-pole low-passes with a per-sample cutoff array (Hz)."""
    out = np.zeros_like(x)
    blk = 256
    z1 = z2 = np.zeros(1)
    for i in range(0, len(x), blk):
        a = 1 - np.exp(-TAU * fc[i] / SR)
        y, z1 = signal.lfilter([a], [1, a - 1], x[i:i + blk], zi=z1)
        out[i:i + blk], z2 = signal.lfilter([a], [1, a - 1], y, zi=z2)
    return out


def tape_stop(x, start, dur):
    """Slow the audio to a halt from `start` (s) over `dur` (s); everything after is silence."""
    i0, n = int(start * SR), int(dur * SR)
    speed = np.linspace(1, 0, n) ** 1.3
    pos = i0 + np.cumsum(speed)
    seg = np.interp(pos, np.arange(len(x)), x) * np.linspace(1, 0.2, n)
    out = x.copy()
    out[i0:i0 + n] = seg[:len(out) - i0]
    out[i0 + n:] = 0
    return out


def stutter(x, start, slice_dur, repeats):
    i0, n = int(start * SR), int(slice_dur * SR)
    out = x.copy()
    sl = fade(x[i0:i0 + n], 0.002, 0.004)
    for k in range(repeats):
        a = i0 + k * n
        out[a:a + n] = sl[:max(0, min(n, len(out) - a))]
    return out


class Track:
    """Stereo buffer with sample-accurate placement."""

    def __init__(self, dur):
        self.buf = np.zeros((int(dur * SR) + SR, 2))

    def add(self, t, x, gain=1.0, pan=0.0):
        i = int(round(t * SR))
        if i < 0:
            x, i = x[-i:], 0
        n = min(len(x), len(self.buf) - i)
        if n <= 0:
            return
        gl, gr = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
        self.buf[i:i + n, 0] += x[:n] * gain * gl * 1.4142
        self.buf[i:i + n, 1] += x[:n] * gain * gr * 1.4142


NOTE = {n: 440.0 * 2 ** ((i - 57) / 12) for i, n in enumerate(
    [f"{p}{o}" for o in range(0, 8) for p in ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")])}


def beat(dur, bpm=130, variant="main", seed=0):
    """Phonk-flavoured loop starting on a downbeat. Returns (drums+bass+bell) stereo of length dur."""
    step = 60.0 / bpm / 4
    tr = Track(dur + 1)
    bars = int(np.ceil(dur / (step * 16)))
    K, Sn, H, HO = kick(), snare(), hat(), hat(open_=True)
    bass_a = [(0, 3, "F1"), (3, 3, "F1"), (6, 2, "Ab1"), (8, 3, "F1"), (11, 3, "Eb1"), (14, 2, "C2")]
    bass_b = [(0, 3, "F1"), (3, 3, "F1"), (6, 2, "Ab1"), (8, 3, "Bb1"), (11, 3, "Ab1"), (14, 2, "G1")]
    bell_a = [(0, "F5"), (2, "F5"), (3, "Ab5"), (6, "F5"), (8, "Eb5"), (10, "F5"), (12, "C6"), (14, "Ab5")]
    bell_b = [(0, "F5"), (2, "F5"), (3, "Ab5"), (6, "Bb5"), (8, "C6"), (10, "Ab5"), (12, "F5"), (14, "Eb5")]
    bell_c = [(0, "C6"), (3, "Ab5"), (6, "F5"), (7, "Ab5"), (8, "Bb5"), (11, "Ab5"), (12, "F5"), (14, "C5")]
    r = np.random.default_rng(seed)
    for b in range(bars):
        t0 = b * step * 16
        odd = b % 2 == 1
        if variant in ("main", "alt"):
            kicks = (0, 3, 8, 11) if odd else (0, 6, 8)
            snares = (4, 12)
            hats = [(s, 0.9 if s % 4 == 0 else 0.5) for s in range(0, 16, 2)] + ([(13, 0.35), (15, 0.4)] if odd else [(7, 0.3)])
            bass = bass_b if odd else bass_a
            bells = (bell_c if odd else bell_b) if variant == "alt" else (bell_b if odd else bell_a)
        elif variant == "sparse":
            kicks, snares = (0, 8), ()
            hats = [(s, 0.6 if s % 4 == 0 else 0.3) for s in range(0, 16, 2)]
            bass = [(0, 6, "F1"), (8, 6, "Ab1" if odd else "Eb1")]
            bells = [(0, "F5"), (10, "Ab5")] if odd else [(0, "F5")]
        else:  # half-time
            kicks, snares = (0, 10), (8,)
            hats = [(s, 0.55) for s in range(0, 16, 4)] + [(14, 0.3)]
            bass = [(0, 7, "F1"), (8, 7, "Db2" if odd else "Eb1")]
            bells = [(0, "F5"), (6, "C5"), (12, "Ab4")] if odd else [(0, "F5"), (8, "Eb5")]
        for s in kicks:
            tr.add(t0 + s * step, K, 0.95)
        for s in snares:
            tr.add(t0 + s * step, Sn, 0.62, pan=0.05)
        for s, v in hats:
            tr.add(t0 + s * step + r.uniform(-0.002, 0.002), H, 0.42 * v, pan=0.25)
        if variant in ("main", "alt") and odd:
            tr.add(t0 + 10 * step, HO, 0.26, pan=-0.25)
        for s, ln, n in bass:
            tr.add(t0 + s * step, bass808(NOTE[n], ln * step * 0.92), 0.70)
        for s, n in bells:
            c = cowbell(NOTE[n])
            tr.add(t0 + s * step, c, 0.50, pan=-0.2)
            tr.add(t0 + (s + 3) * step, c, 0.14, pan=0.5)       # dotted-eighth echo
    return tr.buf[:int(dur * SR)]
