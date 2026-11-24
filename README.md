# Leopardix short-form video pipeline

Vertical videos (TikTok / Shorts / Reels) for Leopardix, produced end to end from code: voiceover, original
voxel background footage, synthesised music and effects, captions and in-frame gags.

## Videos

| Video | Deliverables |
|---|---|
| `videos/german-basement` — "in Germany, in German's basement" | `final.mp4`, `cover.jpg`, `description.txt`, `script.md`, `tts.txt` |

## Rebuilding a video

Needs Python 3.12, `ffmpeg`, `gcc`, and `pip install numpy scipy pillow opencv-python-headless faster-whisper`.

```sh
cd videos/german-basement
export ELEVENLABS_API_KEY=...            # never commit the key
python ../../pipeline/tts.py --text-file tts.txt --speed 1.2 --seed 11 --out vo/take_11   # repeat for 23, 42
python ../../pipeline/vo_comp.py --text tts.txt --out vo --base take_23 --speed 1.10 \
    --gaps "16:0.28,21:0.22,23:0.24,24:0.42,25:0.24" --takes vo/take_11 vo/take_23 vo/take_42
python shots.py        # background footage (resumable; add --budget N to stop after N seconds)
python audio.py        # music + effects + mix
python compose.py      # captions, gags, outro -> build/master.mp4
python finish.py pass1 && python finish.py pass2 && python finish.py qa
```

Every edit decision is cut to the voiceover: `edit.py` derives captions and cue times from
`vo/words.json`, so a new take re-times the whole video. The takes and the comped voiceover are committed, so
the picture and mix can be rebuilt without spending TTS credits.

## Layout

- `pipeline/tts.py`, `vo_comp.py`, `qa_take.py`, `qa_vo.py` — ElevenLabs takes, sentence-level comp, Whisper checks.
- `pipeline/vox/` — small CPU voxel raycaster (`vox.c`), procedural 16×16 textures and world builders. All
  footage is rendered here, so nothing is matched against existing gameplay uploads.
- `pipeline/synth.py` — drums, bass, cowbell and sound effects generated from code.
- `pipeline/gfx.py` — caption/stamp sprites and frame effects.
- `assets/fonts` — Montserrat, Russo One, Monocraft, JetBrains Mono (SIL OFL, licences included).
- `assets/brand` — the Leopardix mark from leopardix.net.
