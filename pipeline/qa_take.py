#!/usr/bin/env python3
"""QA metrics for TTS takes: Whisper transcript check + acoustic liveliness + stress heuristics.

Usage: qa_take.py take1 take2 ...   (expects <take>.mp3 and <take>.json from tts.py)
"""
import json
import re
import subprocess
import sys
import unicodedata

import numpy as np

SR = 16000
VOWELS = "аеёиоуыэюя"
_model = None


def load_audio(path, sr=SR):
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32)


def whisper_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8", cpu_threads=2)
    return _model


def norm_words(text):
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn" or c == "\u0306" or c == "\u0308")
    text = unicodedata.normalize("NFC", text).lower().replace("ё", "е")
    return re.findall(r"[а-яa-z0-9]+", text)


def wer(ref, hyp):
    d = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=int)
    d[:, 0] = np.arange(len(ref) + 1)
    d[0, :] = np.arange(len(hyp) + 1)
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    return d[-1, -1] / max(1, len(ref))


def f0_track(x, sr=SR, fmin=60, fmax=320, hop=0.010, win=0.040):
    n, h = int(win * sr), int(hop * sr)
    lo, hi = int(sr / fmax), int(sr / fmin)
    f0 = []
    for s in range(0, len(x) - n, h):
        fr = x[s:s + n] * np.hanning(n)
        e = float(np.sqrt(np.mean(fr ** 2)))
        if e < 0.01:
            f0.append(0.0)
            continue
        ac = np.correlate(fr, fr, mode="full")[n - 1:]
        ac = ac / (ac[0] + 1e-9)
        k = lo + int(np.argmax(ac[lo:hi]))
        f0.append(sr / k if ac[k] > 0.45 else 0.0)
    return np.array(f0)


def rms_env(x, sr=SR, hop=0.010, win=0.025):
    n, h = int(win * sr), int(hop * sr)
    return np.array([np.sqrt(np.mean(x[s:s + n] ** 2)) for s in range(0, len(x) - n, h)])


def stress_report(meta, x):
    """For words written with a combining acute, check the marked vowel is the longest/loudest."""
    al = meta["alignment"]
    chars, st, en = al["characters"], al["character_start_times_seconds"], al["character_end_times_seconds"]
    out = []
    i = 0
    while i < len(chars):
        if not chars[i].isalpha():
            i += 1
            continue
        j = i
        while j < len(chars) and (chars[j].isalpha() or chars[j] == "\u0301"):
            j += 1
        word = "".join(chars[i:j])
        if "\u0301" in word:
            vows = []
            for k in range(i, j):
                if chars[k].lower() in VOWELS:
                    seg = x[int(st[k] * SR):int(en[k] * SR)]
                    energy = float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0
                    marked = k + 1 < j and chars[k + 1] == "\u0301"
                    vows.append((chars[k], round(en[k] - st[k], 3), round(energy, 3), marked))
            score = [d * e for _, d, e, _ in vows]
            best = int(np.argmax(score)) if score else -1
            ok = bool(vows and vows[best][3])
            out.append(f"{word}: {'OK' if ok else 'CHECK'} {vows}")
        i = j
    return out


def analyze(prefix):
    meta = json.load(open(prefix + ".json", encoding="utf-8"))
    x = load_audio(prefix + ".mp3")
    dur = len(x) / SR
    segs, info = whisper_model().transcribe(
        x, language="ru", word_timestamps=True, beam_size=5, condition_on_previous_text=False,
        vad_filter=False,
    )
    segs = list(segs)
    hyp_text = " ".join(s.text.strip() for s in segs)
    words = [w for s in segs for w in (s.words or [])]
    low = [(w.word.strip(), round(w.probability, 2)) for w in words if w.probability < 0.5]
    ref, hyp = norm_words(meta["text"]), norm_words(hyp_text)
    f0 = f0_track(x)
    v = f0[f0 > 0]
    env = rms_env(x)
    speech = env > 0.02
    semis = 12 * np.log2(v / np.median(v)) if len(v) else np.array([0.0])
    al = meta["alignment"]
    pauses = []
    st, en, ch = al["character_start_times_seconds"], al["character_end_times_seconds"], al["characters"]
    # pauses = long gaps on whitespace/punctuation characters
    for c, a, b in zip(ch, st, en):
        if not c.isalnum() and b - a > 0.2:
            pauses.append(round(b - a, 2))
    print(f"=== {prefix}")
    print(f"  dur={dur:.2f}s words={len(ref)} wpm={len(ref) / dur * 60:.0f} speech_ratio={speech.mean():.2f} pauses>0.2s={len(pauses)} sum={sum(pauses):.2f}s")
    print(f"  whisper: WER={wer(ref, hyp):.3f} avg_logprob={np.mean([s.avg_logprob for s in segs]):.3f} low_conf={low}")
    print(f"  text: {hyp_text}")
    print(f"  pitch: median={np.median(v):.0f}Hz p10-p90 range={np.percentile(semis, 90) - np.percentile(semis, 10):.1f}st std={semis.std():.2f}st")
    print(f"  energy: speech_rms_db={20 * np.log10(env[speech].mean() + 1e-9):.1f} dyn_range_db={20 * np.log10((np.percentile(env[speech], 95) + 1e-9) / (np.percentile(env[speech], 10) + 1e-9)):.1f}")
    for line in stress_report(meta, x):
        print("  stress:", line)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        analyze(p)
