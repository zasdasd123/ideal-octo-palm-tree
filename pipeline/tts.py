#!/usr/bin/env python3
"""ElevenLabs TTS with character-level timestamps.

Writes <out>.mp3 and <out>.json (alignment). The API key is read from
ELEVENLABS_API_KEY and is never written to disk.
"""
import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request

API = "https://api.elevenlabs.io/v1/text-to-speech/{voice}/with-timestamps?output_format=mp3_44100_128"
BRIAN = "nPczCjzI2devNBz1zQrb"


def synth(text, voice, seed, out, model="eleven_multilingual_v2",
          stability=0.30, similarity=0.80, style=0.70, speaker_boost=True, speed=None):
    body = {
        "text": text,
        "model_id": model,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity,
            "style": style,
            "use_speaker_boost": speaker_boost,
        },
    }
    if speed is not None:
        body["voice_settings"]["speed"] = speed
    if seed is not None:
        body["seed"] = seed
    req = urllib.request.Request(
        API.format(voice=voice),
        data=json.dumps(body).encode("utf-8"),
        headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"], "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            payload = json.loads(r.read())
            cost = r.headers.get("character-cost") or r.headers.get("x-character-count")
    except urllib.error.HTTPError as e:
        sys.exit(f"ElevenLabs HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:600]}")
    with open(out + ".mp3", "wb") as f:
        f.write(base64.b64decode(payload["audio_base64"]))
    meta = {
        "text": text, "voice": voice, "model": model, "seed": seed,
        "settings": body["voice_settings"],
        "alignment": payload.get("alignment"),
        "normalized_alignment": payload.get("normalized_alignment"),
    }
    with open(out + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    print(f"{out}.mp3: chars={len(text)} cost_header={cost}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default=BRIAN)
    ap.add_argument("--text")
    ap.add_argument("--text-file")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="eleven_multilingual_v2")
    ap.add_argument("--speed", type=float, help="native TTS pace, 0.7-1.2")
    a = ap.parse_args()
    txt = a.text if a.text else open(a.text_file, encoding="utf-8").read().strip()
    synth(txt, a.voice, a.seed, a.out, model=a.model, speed=a.speed)
