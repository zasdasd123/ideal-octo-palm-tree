#!/usr/bin/env python3
"""Comp the voiceover from several TTS takes and post-process it.

Per sentence it picks the liveliest take that Whisper transcribes cleanly, caps
silences at 0.2 s, speeds the result up (tempo only, no pitch shift), then applies
EQ / compression / loudness normalisation. Word timings on the final timeline are
written next to the audio so captions and gags can be cut to the voice.

Usage: vo_comp.py --text tts.txt --out vo --takes vo/take_11 vo/take_23 ... [--base take_23]
"""
import argparse
import json
import os
import re
import subprocess
import unicodedata

import numpy as np

SR = 44100
HOP = 220  # 5 ms
MAX_SIL = 0.20
SPEED = 1.09
EQUIV = [
    ("двести сорок", "240"), ("200 сорок", "240"), ("восьмидесяти процентов", "80"),
    ("двадцать четыре", "24"), ("пять", "5"), ("тэ пэ эс", "тпс"), ("тп с", "тпс"),
    ("дата центр", "датацентр"), ("турбо буст", "турбобуст"), ("сдохла", "сдохло"),
]


def load(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def env_db(x, win=882):
    pad = np.concatenate([x, np.zeros(win, dtype=np.float32)])
    c = np.concatenate([[0.0], np.cumsum(pad.astype(np.float64) ** 2)])
    idx = np.arange(0, len(x), HOP)
    return 10 * np.log10((c[idx + win] - c[idx]) / win + 1e-12)


def strip_marks(s):
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if c not in "\u0301")
    return unicodedata.normalize("NFC", s)


def norm(text):
    t = strip_marks(text).lower().replace("ё", "е")
    t = " ".join(re.findall(r"[а-яa-z0-9]+", t))
    for a, b in EQUIV:
        t = t.replace(a, b)
    return t.split()


def edit_distance(a, b):
    d = np.zeros((len(a) + 1, len(b) + 1), dtype=int)
    d[:, 0] = np.arange(len(a) + 1)
    d[0, :] = np.arange(len(b) + 1)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
    return int(d[-1, -1])


def sentences(text):
    """[(char_start, char_end, [(word, c0, c1), ...])] over the exact TTS text."""
    out, start = [], 0
    for m in re.finditer(r"[.!?]+[»\"]?(?=\s|$)", text):
        if text[m.start():m.end()].startswith("...") and m.end() < len(text) and text[m.end():].lstrip()[:1].islower():
            continue  # ellipsis inside a sentence
        seg = (start, m.end())
        words = [(w.group(0), w.start(), w.end()) for w in re.finditer(r"[^\s—«»\"]*[0-9A-Za-zА-Яа-яЁё\u0301][^\s—«»\"]*", text[seg[0]:seg[1]])]
        words = [(w, a + seg[0], b + seg[0]) for w, a, b in words]
        if words:
            out.append((seg[0], seg[1], words))
        start = m.end()
    return out


def f0_range(x):
    n, h = int(0.04 * SR), int(0.01 * SR)
    lo, hi = SR // 320, SR // 60
    vals = []
    for s in range(0, len(x) - n, h):
        fr = x[s:s + n] * np.hanning(n)
        if np.sqrt(np.mean(fr ** 2)) < 0.01:
            continue
        spec = np.fft.rfft(fr, 2 * n)
        ac = np.fft.irfft(np.abs(spec) ** 2)[:n]
        ac /= ac[0] + 1e-9
        k = lo + int(np.argmax(ac[lo:hi]))
        if ac[k] > 0.45:
            vals.append(SR / k)
    if len(vals) < 5:
        return 0.0
    st = 12 * np.log2(np.array(vals) / np.median(vals))
    return float(np.percentile(st, 90) - np.percentile(st, 10))


class Take:
    def __init__(self, prefix, text, sents, whisper):
        self.name = os.path.basename(prefix)
        meta = json.load(open(prefix + ".json", encoding="utf-8"))
        assert strip_marks(meta["text"]) == strip_marks(text), "take was generated from a different text"
        al = meta["alignment"]
        assert len(al["characters"]) == len(meta["text"]) == len(text)
        self.cs = np.array(al["character_start_times_seconds"])
        self.ce = np.array(al["character_end_times_seconds"])
        self.x = load(prefix + ".mp3")
        self.env = env_db(self.x)
        self.thr = float(np.percentile(self.env, 95)) - 36.0
        self.sents = sents
        # cut points between sentences: centre of the quietest stretch in the alignment gap
        cuts = [0.0]
        for (a0, a1, w0), (b0, b1, w1) in zip(sents[:-1], sents[1:]):
            ta, tb = self.ce[w0[-1][2] - 1], self.cs[w1[0][1]]
            i0, i1 = int(max(0, ta - 0.03) * SR / HOP), int((tb + 0.03) * SR / HOP) + 1
            seg = self.env[i0:i1]
            quiet = seg < self.thr
            if quiet.any():
                best, cur, run0 = (0, 0), 0, 0
                for k, q in enumerate(np.append(quiet, False)):
                    if q and cur == 0:
                        run0 = k
                    cur = cur + 1 if q else 0
                    if cur > best[1] - best[0]:
                        best = (run0, run0 + cur)
                k = (best[0] + best[1]) // 2
            else:
                k = int(np.argmin(seg))
            cuts.append((i0 + k) * HOP / SR)
        cuts.append(len(self.x) / SR)
        self.cuts = cuts
        self.words_whisper = whisper

    def clip(self, i):
        """Trimmed audio for sentence i plus word times relative to the clip."""
        t0, t1 = self.cuts[i], self.cuts[i + 1]
        i0, i1 = int(t0 * SR / HOP), int(t1 * SR / HOP)
        loud = np.where(self.env[i0:i1] > self.thr)[0]
        a = max(t0, (i0 + loud[0]) * HOP / SR - 0.030)
        b = min(t1, (i0 + loud[-1]) * HOP / SR + 0.020 + 0.080)
        x = self.x[int(a * SR):int(b * SR)].copy()
        fade = int(0.008 * SR)
        x[:fade] *= np.linspace(0, 1, fade)
        x[-fade:] *= np.linspace(1, 0, fade)
        words = [(w, float(self.cs[c0] - a), float(self.ce[c1 - 1] - a)) for w, c0, c1 in self.sents[i][2]]
        return x, words, (a, b)

    def whisper_errors(self, i):
        a, b = self.cuts[i], self.cuts[i + 1]
        hyp = " ".join(w for w, s, e in self.words_whisper if a <= (s + e) / 2 < b)
        ref = " ".join(w for w, _, _ in self.sents[i][2])
        return edit_distance(norm(ref), norm(hyp)), hyp


def compress_silence(x, words, thr):
    """Cap silences inside a clip at MAX_SIL; remap word times."""
    env = env_db(x)
    quiet = env < thr
    cuts, k = [], 0
    while k < len(quiet):
        if quiet[k]:
            j = k
            while j < len(quiet) and quiet[j]:
                j += 1
            dur = (j - k) * HOP / SR
            if dur > MAX_SIL and k > 0 and j < len(quiet):
                mid = (k + j) / 2 * HOP / SR
                cuts.append((mid - (dur - MAX_SIL) / 2, mid + (dur - MAX_SIL) / 2))
            k = j
        else:
            k += 1
    if not cuts:
        return x, words, 0.0
    parts, pos, xf = [], 0, int(0.005 * SR)
    for c0, c1 in cuts:
        seg = x[pos:int(c0 * SR)].copy()
        seg[-xf:] *= np.linspace(1, 0, xf)
        parts.append(seg)
        pos = int(c1 * SR)
        x[pos:pos + xf] *= np.linspace(0, 1, xf)
    parts.append(x[pos:])

    def remap(t):
        return t - sum(min(max(t - c0, 0.0), c1 - c0) for c0, c1 in cuts)
    return np.concatenate(parts), [(w, remap(s), remap(e)) for w, s, e in words], sum(c1 - c0 for c0, c1 in cuts)


def whisper_words(prefix, cache):
    path = os.path.join(cache, os.path.basename(prefix) + ".whisper.json")
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    from faster_whisper import WhisperModel
    model = whisper_words.model = getattr(whisper_words, "model", None) or WhisperModel(
        "large-v3-turbo", device="cpu", compute_type="int8", cpu_threads=2)
    path16 = prefix if prefix.endswith(".wav") else prefix + ".mp3"
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", path16, "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    segs, _ = model.transcribe(np.frombuffer(pcm, dtype=np.float32), language="ru",
                               word_timestamps=True, beam_size=5, condition_on_previous_text=False)
    words = [(w.word.strip(), float(w.start), float(w.end)) for s in segs for w in (s.words or [])]
    json.dump(words, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    return words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--takes", nargs="+", required=True)
    ap.add_argument("--base", help="take to prefer when scores are close")
    ap.add_argument("--base-bonus", type=float, default=1.5)
    ap.add_argument("--gap", type=float, default=0.15, help="default pause between sentences, seconds")
    ap.add_argument("--skip", default="", help="comma-separated sentence numbers (1-based) to drop")
    ap.add_argument("--gaps", default="", help="editorial pauses before a sentence, e.g. 24:0.45,25:0.3")
    ap.add_argument("--force", default="", help="force a take for a sentence, e.g. 7:take_23")
    ap.add_argument("--speed", type=float, default=SPEED)
    a = ap.parse_args()
    text = open(a.text, encoding="utf-8").read().strip()
    sents = sentences(text)
    takes = [Take(p, text, sents, whisper_words(p, a.out)) for p in a.takes]
    skip = {int(s) for s in a.skip.split(",") if s}
    gaps = {int(k): float(v) for k, v in (g.split(":") for g in a.gaps.split(",") if g)}
    force = dict(f.split(":") for f in a.force.split(",") if f)

    rows = []
    for i in range(len(sents)):
        cand = []
        for t in takes:
            x, words, _ = t.clip(i)
            errs, hyp = t.whisper_errors(i)
            cand.append({"take": t, "x": x, "words": words, "errs": errs, "hyp": hyp, "dur": len(x) / SR,
                         "range": f0_range(x), "rms": float(10 * np.log10(np.mean(x ** 2) + 1e-12))})
        rng = np.array([c["range"] for c in cand])
        rms = np.array([c["rms"] for c in cand])
        for c, r, e in zip(cand, (rng - rng.mean()) / (rng.std() + 1e-6), (rms - rms.mean()) / (rms.std() + 1e-6)):
            c["score"] = float(r + 0.7 * e - 3.0 * c["errs"] + (a.base_bonus if c["take"].name == a.base else 0.0))
        best = max(cand, key=lambda c: c["score"])
        if str(i + 1) in force:
            best = next(c for c in cand if c["take"].name == force[str(i + 1)])
        rows.append((i, cand, best))

    out, words_out, pos, report = [], [], 0.0, []
    for i, cand, best in rows:
        ref = text[sents[i][0]:sents[i][1]].strip()
        line = " | ".join(f"{c['take'].name[-2:]}:{c['dur']:.2f}s r{c['range']:.1f} {c['rms']:.0f}dB e{c['errs']} s{c['score']:+.1f}" for c in cand)
        mark = "SKIP" if i + 1 in skip else best["take"].name
        report.append(f"S{i + 1:02d} [{mark}] {strip_marks(ref)[:46]:46} {line}")
        for c in cand:
            if c["errs"]:
                report.append(f"      whisper({c['take'].name}): {c['hyp']}")
        if i + 1 in skip:
            continue
        x, words, saved = compress_silence(best["x"].copy(), best["words"], best["take"].thr)
        gap = gaps.get(i + 1, a.gap) if out else 0.0
        out.append(np.zeros(int(gap * SR), dtype=np.float32))
        pos += len(out[-1]) / SR
        for w, s, e in words:
            words_out.append({"w": strip_marks(w), "s": pos + s, "e": pos + e, "sent": i + 1})
        out.append(x)
        pos += len(x) / SR
    comp = np.concatenate(out + [np.zeros(int(0.25 * SR), dtype=np.float32)])
    os.makedirs(a.out, exist_ok=True)
    raw = os.path.join(a.out, "vo_comp_raw.wav")
    final = os.path.join(a.out, "vo.wav")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-", raw],
                   input=comp.astype(np.float32).tobytes(), check=True)
    # Tempo only (no pitch shift), +3 dB bass shelf at 120 Hz, light presence, compression.
    chain = (f"atempo={a.speed},highpass=f=70,equalizer=f=120:t=q:w=0.9:g=3,equalizer=f=4200:t=q:w=1.2:g=1.5,"
             "acompressor=threshold=-21dB:ratio=3.5:attack=5:release=90:makeup=3")
    # Static gain to -16 LUFS: single-pass loudnorm would ride the level and pump between phrases.
    probe = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", raw, "-af", chain + ",ebur128", "-f", "null", "-"],
                           capture_output=True, text=True).stderr
    measured = float(re.findall(r"I:\s+(-?[0-9.]+) LUFS", probe)[-1])
    chain += f",volume={-16.0 - measured:.2f}dB,alimiter=limit=0.84:attack=3:release=60:level=false,aresample=48000"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", raw, "-af", chain, "-ar", "48000", "-ac", "1", final], check=True)
    for w in words_out:
        w["s"] = round(w["s"] / a.speed, 3)
        w["e"] = round(w["e"] / a.speed, 3)
    dur = len(comp) / SR / a.speed
    json.dump({"duration": round(dur, 3), "speed": a.speed, "words": words_out,
               "picks": {f"S{i + 1}": best["take"].name for i, _, best in rows if i + 1 not in skip}},
              open(os.path.join(a.out, "words.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n".join(report))
    n = len(words_out)
    print(f"comp: {len(comp) / SR:.2f}s raw -> {dur:.2f}s at x{a.speed}; words={n}; wpm={n / dur * 60:.0f}")


if __name__ == "__main__":
    main()
