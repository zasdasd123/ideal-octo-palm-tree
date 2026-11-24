#!/usr/bin/env python3
"""Check the final voiceover: Whisper transcript vs script, and caption timing drift.

Usage: qa_vo.py vo/vo.wav vo/words.json
"""
import json
import re
import subprocess
import sys

import numpy as np

wav, words_path = sys.argv[1], sys.argv[2]
ref = json.load(open(words_path, encoding="utf-8"))
pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", wav, "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
                     capture_output=True, check=True).stdout
from faster_whisper import WhisperModel  # noqa: E402

model = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8", cpu_threads=2)
segs, _ = model.transcribe(np.frombuffer(pcm, dtype=np.float32), language="ru", word_timestamps=True,
                           beam_size=5, condition_on_previous_text=False)
hyp = [(w.word.strip(), w.start, w.end) for s in segs for w in (s.words or [])]
print("WHISPER:", " ".join(w for w, _, _ in hyp))


def key(w):
    return re.sub(r"[^а-яa-z0-9]", "", w.lower().replace("ё", "е"))


# greedy in-order match of identical words to measure timing drift
drift, j = [], 0
for r in ref["words"]:
    k = key(r["w"])
    for jj in range(j, min(j + 4, len(hyp))):
        if key(hyp[jj][0]) == k:
            drift.append((r["w"], r["s"] - hyp[jj][1]))
            j = jj + 1
            break
d = np.array([x for _, x in drift])
print(f"matched {len(d)}/{len(ref['words'])} words; start drift (ours - whisper): mean={d.mean() * 1000:.0f}ms "
      f"p10={np.percentile(d, 10) * 1000:.0f}ms p90={np.percentile(d, 90) * 1000:.0f}ms max|d|={np.abs(d).max() * 1000:.0f}ms")
print("worst:", sorted(drift, key=lambda x: -abs(x[1]))[:6])
print(f"duration={ref['duration']}s words={len(ref['words'])} wpm={len(ref['words']) / ref['duration'] * 60:.0f}")

# Sentence-initial words should start right at an energy onset (Whisper is early across pauses).
x = np.frombuffer(pcm, dtype=np.float32)
env = np.array([np.sqrt(np.mean(x[i:i + 320] ** 2)) for i in range(0, len(x) - 320, 80)])  # 5 ms hop
thr = np.percentile(env, 95) * 10 ** (-30 / 20)
onsets, prev = [], 0
for w in ref["words"]:
    if w["sent"] != prev:
        prev = w["sent"]
        i = int(w["s"] / 0.005)
        lo = max(0, i - 60)
        quiet = np.where(env[lo:i + 40] < thr)[0]
        if len(quiet):
            k = lo + quiet[quiet <= min(60, len(quiet) and quiet.max())].max() if (quiet <= 60).any() else lo + quiet.min()
            onsets.append((w["w"], round(w["s"] - (k + 1) * 0.005, 3)))
od = np.array([d for _, d in onsets])
print(f"sentence-start vs energy onset: mean={od.mean() * 1000:.0f}ms min={od.min() * 1000:.0f}ms max={od.max() * 1000:.0f}ms")
print("  ", onsets)
