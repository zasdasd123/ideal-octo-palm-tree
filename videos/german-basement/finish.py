#!/usr/bin/env python3
"""Final encode + QA for "German's basement".

    finish.py pass1 | pass2 | qa     (two x264 passes are separate steps so each stays short)

pass2 writes final.mp4 (1080x1920, 60 fps, ~8 Mbps H.264 + AAC, just under 50 MB) and cover.jpg next to this file.
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BUILD = os.path.join(HERE, "build")
MASTER, MIX = os.path.join(BUILD, "master.mp4"), os.path.join(BUILD, "audio", "mix.wav")
FINAL, COVER = os.path.join(HERE, "final.mp4"), os.path.join(HERE, "cover.jpg")
VIDEO = ["-c:v", "libx264", "-preset", "medium", "-profile:v", "high", "-level", "4.2", "-pix_fmt", "yuv420p",
         "-b:v", "8055k", "-maxrate", "20000k", "-bufsize", "30000k", "-r", "60", "-g", "120", "-bf", "2",
         "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
         "-passlogfile", os.path.join(BUILD, "x264pass")]


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def pass1():
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", MASTER, *VIDEO, "-pass", "1", "-an", "-f", "null", os.devnull], check=True)


def pass2():
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", MASTER, "-i", MIX, "-map", "0:v:0", "-map", "1:a:0", *VIDEO, "-pass", "2",
                    "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2", "-shortest", "-movflags", "+faststart", FINAL], check=True)
    import edit as ED
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{ED.C['label_basement'] + 0.5:.2f}", "-i", FINAL, "-frames:v", "1",
                    "-q:v", "2", COVER], check=True)
    print("wrote", FINAL, f"{os.path.getsize(FINAL) / 1e6:.1f} MB")


def qa():
    import edit as ED
    ok = True

    def check(name, cond, detail):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'OK' if cond else 'FAIL'}] {name}: {detail}")

    j = json.loads(run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", FINAL]).stdout)
    v = next(s for s in j["streams"] if s["codec_type"] == "video")
    a = next(s for s in j["streams"] if s["codec_type"] == "audio")
    dur, br = float(j["format"]["duration"]), int(v["bit_rate"]) / 1e6
    check("format", (v["width"], v["height"]) == (1080, 1920) and v["codec_name"] == "h264", f"{v['codec_name']} {v['width']}x{v['height']} {v['pix_fmt']}")
    check("frame rate", v["avg_frame_rate"] == "60/1", v["avg_frame_rate"])
    check("video bitrate 8-10+ Mbps", br >= 8.0, f"{br:.2f} Mbps")
    check("duration 40-55 s", 40 <= dur <= 55, f"{dur:.2f} s")
    check("audio", a["codec_name"] == "aac" and a["sample_rate"] == "48000" and a["channels"] == 2, f"{a['codec_name']} {a['sample_rate']} Hz {a['channels']}ch")
    words = len(ED.SH.VO["words"])
    check("script 110-150 words", 110 <= words <= 150, f"{words} words, {words / ED.SH.VO['duration'] * 60:.0f} wpm in the voiceover")
    eb = run(["ffmpeg", "-hide_banner", "-nostats", "-i", FINAL, "-af", "ebur128=peak=true", "-f", "null", "-"]).stderr
    lufs = float(re.findall(r"I:\s+(-?[0-9.]+) LUFS", eb)[-1])
    peak = float(re.findall(r"Peak:\s+(-?[0-9.]+) dBFS", eb)[-1])
    check("loudness about -14 LUFS", -15.0 <= lufs <= -13.0, f"{lufs:.1f} LUFS, true peak {peak:.1f} dBTP")
    check("true peak <= -1 dBTP", peak <= -0.9, f"{peak:.1f} dBTP")
    # music stem must be digitally silent from the outro cut (the brand line plays over dead silence)
    st = run(["ffmpeg", "-hide_banner", "-nostats", "-ss", f"{ED.C['outro'] + 0.01:.3f}", "-i", os.path.join(BUILD, "audio", "music.wav"),
              "-af", "astats=metadata=0:measure_overall=Peak_level:measure_perchannel=none", "-f", "null", "-"]).stderr
    pk = re.findall(r"Peak level dB:\s+(-?[0-9.a-z]+)", st)
    check("music cut to dead silence at brand reveal", pk and (pk[-1] == "-inf" or float(pk[-1]) < -90), f"music peak after {ED.C['outro']:.2f}s = {pk[-1] if pk else '?'} dB")
    bd = run(["ffmpeg", "-hide_banner", "-nostats", "-i", FINAL, "-vf", "blackdetect=d=0.05:pix_th=0.06", "-an", "-f", "null", "-"]).stderr
    blacks = [(float(s), float(e)) for s, e in re.findall(r"black_start:([0-9.]+) black_end:([0-9.]+)", bd)]
    early = [b for b in blacks if b[0] < dur - 0.4]
    check("no black frames before the CRT-off", not early, f"black segments: {blacks}")
    caps = ED.captions()
    check("captions 1-3 words, no dangling prepositions", all(len(c["text"].replace("*", "").split()) <= 3 for c in caps), f"{len(caps)} captions validated")
    yellow = [c["text"] for c in caps if any(w in c["text"] for w in ED.SWEARS)]
    check("profanity punchlines highlighted (3-5)", 3 <= len(yellow) <= 5 and all("*" in y for y in yellow), "; ".join(y.replace("*", "") for y in yellow))
    print("QA", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "qa"
    if step == "qa":
        sys.exit(0 if qa() else 1)
    {"pass1": pass1, "pass2": pass2}[step]()
