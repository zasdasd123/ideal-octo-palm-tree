#!/usr/bin/env python3
"""Compositor for "German's basement": background shots + effects + captions + in-frame gags + outro.

    compose.py                       -> build/master.mp4 (video only, near-lossless)
    compose.py --still 12.5 out.png  -> one composed frame
    compose.py --bg build/preview    -> use the low-res animatic as background (fast iteration)
"""
import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
import edit as ED  # noqa: E402
from pipeline import gfx as G  # noqa: E402

W_, H_, FPS = 1080, 1920, ED.FPS
C, CUT, WT, ET, END = ED.C, ED.CUT, ED.W, ED.E, ED.END
N = int(round(END * FPS))
CAP_Y, CAP_W = 1190, 860
HUD_X, HUD_Y = 72, 236


def smooth(u):
    u = np.clip(u, 0, 1)
    return u * u * (3 - 2 * u)


def keyed(age, keys):
    """Per-frame keyframes (age in frames); holds the last value."""
    i = int(age)
    return keys[i] if i < len(keys) else keys[-1]


SLAM = [2.3, 1.7, 1.25, 0.96, 0.99, 1.0]
POP = [0.78, 0.95, 1.05, 1.06, 1.02, 1.0]
POP_STRONG = [1.42, 1.18, 1.0, 0.97, 1.0]

# ------------------------------------------------------------------ timed effect events

PUNCH, SHAKE, FLASH = [], [], []     # (t, amplitude, time constant)
CAPS = ED.captions()
for c in CAPS:
    if c["strong"]:
        PUNCH.append((c["start"] + 0.03, 0.045, 0.15))
        SHAKE.append((c["start"] + 0.03, 6, 0.10))
    elif "*" in c["text"]:
        PUNCH.append((c["start"] + 0.03, 0.02, 0.13))
for ts, _ in ED.SWEAR_T:                                 # the shouted words get a camera kick
    PUNCH.append((ts, 0.035, 0.15))
    SHAKE.append((ts, 10, 0.12))
for key in ("label_basement", "turbo", "cool", "closed", "positive"):
    SHAKE.append((C[key] + 0.05, 18, 0.14))
    PUNCH.append((C[key] + 0.05, 0.05, 0.18))
    FLASH.append((C[key] + 0.05, 0.14, 0.05))
SHAKE += [(CUT["reveal"], 12, 0.15), (CUT["space"], 20, 0.30), (C["freeze2"], 14, 0.12), (C["badge"] + 0.06, 12, 0.12)]
PUNCH += [(CUT["head"], 0.06, 0.2), (CUT["space"], 0.08, 0.3)]
FLASH += [(CUT["outro"], 0.92, 0.10), (CUT["space"], 0.5, 0.07), (CUT["support"], 0.3, 0.06)]
GLITCH = [(CUT["reveal"] - 0.07, CUT["reveal"] + 0.08, 1.0), (C["freeze1"] - 0.02, C["freeze1"] + 0.07, 0.6),
          (CUT["tps"] - 0.03, CUT["tps"] + 0.09, 0.8), (C["freeze2"], C["freeze2"] + 0.08, 0.7),
          (C["dead"] - 0.03, C["dead"] + 0.06, 0.6)]


def ev(events, t):
    return sum(a * np.exp(-(t - t0) / tau) for t0, a, tau in events if 0 <= t - t0 < tau * 6)


def spin_g(t):
    return float(smooth((t - (C["spin"] - 0.05)) / 1.9)) if CUT["spin"] <= t < CUT["space"] else 0.0


# ------------------------------------------------------------------ sprites

SPR = {}


def spr(key, make):
    if key not in SPR:
        SPR[key] = make()
    return SPR[key]


def stamp(frame, t, t0, sprite, cx, cy, angle, t1=None, fade=0.0):
    if t < t0:
        return
    age = (t - t0) * FPS
    a = min(1.0, 0.45 + age * 0.4)
    if t1 is not None and fade:
        a *= float(np.clip((t1 - t) / fade, 0, 1))
    jit = np.random.default_rng(int(t * FPS)).uniform(-1, 1, 2) * (1.2 if age > 6 else 0)
    G.blit(frame, sprite, cx + jit[0], cy + jit[1], scale=keyed(age, SLAM), angle=angle, alpha=a)


def hud(frame, text, color, y=HUD_Y, x=HUD_X, size=66, alpha=1.0, extra=0):
    """Game-style readout on a translucent dark plate (keeps it legible over bright sky)."""
    s = G.mc_text(text, size, color)
    h, w = s.shape[:2]
    roi = frame[y - 10:y + h + 2, x - 16:x + w + 14 + extra]
    roi[:] = cv2.convertScaleAbs(roi, alpha=1 - 0.6 * alpha)
    G.blit_tl(frame, s, x, y, alpha)
    return w


def typed(text, t, t0, cps=60):
    return text[:max(0, int((t - t0) * cps))]


# ------------------------------------------------------------------ gag layers: (start, end, draw(frame, t))

def tag_germany(frame, t):
    def make():
        img = Image.new("RGBA", (1000, 330), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rectangle((0, 20, 11, 300), fill=G.GOLD + (255,))
        for k, col in enumerate(((20, 20, 20), (221, 0, 0), (255, 206, 0))):       # flag, drawn (not an emoji)
            d.rectangle((40, 30 + k * 16, 118, 46 + k * 16), fill=col + (255,))
        d.rectangle((38, 28, 120, 80), outline=(0, 0, 0, 255), width=3)
        canvas = G.premult(np.array(img))
        G.paste(canvas, G.text_sprite("ЭЛИТНЫЙ ДАТА-ЦЕНТР", 46, name="bold", stroke=6), 120, 6)
        G.paste(canvas, G.text_sprite("ГЕРМАНИЯ", 132), 14, 78)
        return canvas
    s = spr("tag_de", make)
    u = float(smooth((t - C["label_de"]) / 0.22))
    G.blit_tl(frame, s, 70 - 260 * (1 - u), 250, alpha=u)


def label_basement(frame, t):
    s = spr("haz_basement", lambda: G.hazard_sprite(["ПОДВАЛ", "ГЕРМАНА"], 118))
    stamp(frame, t, C["label_basement"], s, 540, 480, -5)


def hud_ping(frame, t):
    u = float(smooth((t - C["lag"]) / max(C["freeze1"] - C["lag"], 0.01)))
    v = int(round(38 + 202 * u))
    col = G.GREEN if v < 80 else (G.YELLOW if v < 150 else G.RED)
    a = float(np.clip((t - (C["lag"] - 0.3)) / 0.12, 0, 1))
    w = hud(frame, f"ПИНГ: {v} мс", col, alpha=a, extra=150)
    bars = max(1, 5 - v // 50)
    for k in range(5):
        x0, hgt = HUD_X + w + 16 + k * 24, 14 + k * 11
        color = (col if k < bars else (70, 70, 70))
        cv2.rectangle(frame, (x0 + 4, HUD_Y + 62 - hgt + 4), (x0 + 20, HUD_Y + 66), (20, 20, 20), -1)
        cv2.rectangle(frame, (x0, HUD_Y + 58 - hgt), (x0 + 16, HUD_Y + 62), color, -1)


TRACE = [("trace0", 0.0, "> tracert Игорь", G.WHITE), ("trace0", 0.28, " 1  подвал Германа      1 мс", G.GREEN),
         ("trace1", 0.0, " 2  бетонная плита     80 мс", G.YELLOW), ("trace2", 0.0, " 3  Аргентина         159 мс", G.RED),
         ("trace3", 0.0, "    ИТОГО: 240 мс = НОРМА", G.YELLOW)]


def trace(frame, t):
    panel = spr("trace_panel", lambda: G.panel_sprite(820, 352, bar=G.GREEN))
    u = float(smooth((t - (C["trace0"] - 0.15)) / 0.15))
    y0 = 226 - 60 * (1 - u)
    G.blit_tl(frame, panel, 66, y0, alpha=u)
    fnt = G.font("mono", 40)
    for k, (cue, off, text, col) in enumerate(TRACE):
        txt = typed(text, t, C[cue] + off, 70)
        if txt.strip():
            s = G.mc_text(txt, 40, col, name="mono")
            G.blit_tl(frame, s, 100, y0 + 30 + k * 60)
    del fnt


def hud_spin(frame, t):
    u = float(smooth((t - C["spin"]) / max(C["turbo"] - C["spin"], 0.01)))
    rpm = int(round(1200 * u / 10) * 10) if t < C["turbo"] else 1200 + int(np.random.default_rng(int(t * 30)).integers(0, 90))
    col = G.WHITE if rpm < 600 else (G.YELLOW if rpm < 1100 else G.RED)
    hud(frame, f"ОТЖИМ: {rpm} об/мин", col)
    s = spr("haz_turbo", lambda: G.hazard_sprite(["ТУРБОБУСТ"], 120))
    stamp(frame, t, C["turbo"], s, 540, 560, -4)


def hud_space(frame, t):
    u = (t - CUT["space"]) / (CUT["tps"] - CUT["space"])
    ghz = 5.2 * np.exp(5.6 * u)
    hud(frame, f"ЧАСТОТА: {ghz:.1f} ГГц", G.YELLOW if ghz < 100 else G.RED)


def hud_tps(frame, t):
    u = float(np.clip((t - CUT["tps"]) / (C["freeze2"] - CUT["tps"]), 0, 1))
    v = 20.0 * (1 - u) ** 1.4
    if t >= C["freeze2"]:
        v = 0.0
        if int((t - C["freeze2"]) * 8) % 2:
            return
    col = G.GREEN if v >= 18 else (G.YELLOW if v >= 12 else G.RED)
    hud(frame, f"ТПС: {v:.1f}", col, size=76)


def hud_humidity(frame, t):
    v = float(np.interp(t, [CUT["flood_a"], WT(13, 4), CUT["flood_b"], CUT["under"], CUT["under"] + 0.6], [78, 80, 87, 97, 100]))
    k = (v - 78) / 22
    col = tuple(int(a * (1 - k) + b * k) for a, b in zip(G.WHITE, G.CYAN))
    hud(frame, f"ВЛАЖНОСТЬ: {int(v)}%", col)
    x0, y0, bw = HUD_X + 4, HUD_Y + 84, 470
    cv2.rectangle(frame, (x0 - 4, y0 - 4), (x0 + bw + 4, y0 + 22), (16, 18, 22), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + int(bw * np.clip((v - 60) / 40, 0, 1)), y0 + 18), col, -1)
    s = spr("st_cool", lambda: G.stamp_sprite(["ЖИДКОСТНОЕ", "ОХЛАЖДЕНИЕ"], G.CYAN, 96, seed=4, sub="ВКЛЮЧЕНО"))
    end = CUT["under"] + 0.6
    if t < end:
        stamp(frame, t, C["cool"], s, 540, 660, -6, t1=end, fade=0.15)


def bubbles(frame, t):
    r = np.random.default_rng(77)
    ov = frame.copy()
    for k in range(26):
        x, sp, ph, sz = r.uniform(40, 1040), r.uniform(260, 620), r.uniform(0, 2400), int(r.integers(8, 22))
        y = 2000 - ((t - CUT["under"]) * sp + ph) % 2200
        xx = x + 18 * np.sin(t * 3 + k)
        cv2.rectangle(ov, (int(xx), int(y)), (int(xx) + sz, int(y) + sz), (214, 240, 248), 3)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, dst=frame)


def ticket_strip(frame, t, t0, text, key):
    s = spr(key, lambda: _strip(text))
    u = float(smooth((t - t0) / 0.14))
    G.blit_tl(frame, s, 66 - 200 * (1 - u), 228, alpha=u)


def _strip(text):
    txt = G.mc_text(text, 50, G.WHITE)
    return G.paste(G.panel_sprite(txt.shape[1] + 70, 92, bar=G.RED), txt, 40, 16)


def ticket(frame, t):
    ticket_strip(frame, t, C["ticket"] - 0.05, "ТИКЕТ #2407", "strip1")
    stamp(frame, t, C["closed"], spr("st_closed", lambda: G.stamp_sprite(["ЗАКРЫТО"], G.RED, 150, seed=2)), 540, 560, -10)
    stamp(frame, t, C["positive"], spr("st_pos", lambda: G.stamp_sprite(["ПОЛОЖИТЕЛЬНЫЙ", "ОТЗЫВ"], G.GREEN, 88, seed=3)), 540, 880, 5)


def igor_ticket(frame, t):
    ticket_strip(frame, t, C["igor_ticket"] - 0.05, "ТИКЕТ #2408 · ИГОРЬ", "strip2")
    panel = spr("msg_panel", lambda: G.panel_sprite(900, 150, bar=None))
    a = float(smooth((t - (C["typed0"] - 0.2)) / 0.15))
    G.blit_tl(frame, panel, 66, 340, alpha=a)
    msg = "«Вы там охуели!»"
    n = int(np.clip((t - C["typed0"]) / (C["typed1"] - C["typed0"]), 0, 1) * len(msg))
    cur = "█" if (n < len(msg) or int(t * 4) % 2 == 0) and t > C["typed0"] - 0.2 else ""
    if n or cur:
        G.blit_tl(frame, G.mc_text(msg[:n] + cur, 62, G.WHITE, name="mono"), 100, 372)
    if t >= C["noq"]:
        G.blit_tl(frame, spr("noq_panel", lambda: G.panel_sprite(900, 76, bar=G.RED)), 66, 498)
        G.blit_tl(frame, G.mc_text(typed("поиск «?» ... 0 совпадений", t, C["noq"], 60), 44, G.RED, name="mono"), 100, 508)
    star = spr("star", lambda: G.star_sprite(150))
    for k in range(5):
        tk = C["stars"] + k * 0.11
        if t >= tk:
            age = (t - tk) * FPS
            G.blit(frame, star, 540 + (k - 2) * 168, 720, scale=keyed(age, [1.8, 1.3, 0.95, 1.0]))
    if t >= C["stars"] + 0.5:
        s = G.mc_text("ОЦЕНКА: 5/5", 56, (255, 196, 48))
        G.blit(frame, s, 540, 850)


def death(frame, t):
    red = np.full_like(frame, (150, 8, 8))
    cv2.addWeighted(frame, 0.52, red, 0.48, 0, dst=frame)
    age = (t - C["dead"]) * FPS
    G.blit(frame, spr("dead", lambda: G.mc_text("ВЫ ПОГИБЛИ!", 112, G.WHITE)), 540, 700, scale=keyed(age, [1.25, 1.08, 1.0]))
    G.blit(frame, spr("score", lambda: G.mc_text("Счёт: 0", 58, (255, 255, 120))), 540, 850)


# ------------------------------------------------------------------ outro

def _logo():
    im = Image.open(os.path.join(G.ROOT, "assets", "brand", "leopardix-mark.png")).convert("RGBA").resize((410, 410), Image.LANCZOS)
    return G.premult(np.array(im))


def _badge():
    w, h = 720, 372
    img = Image.new("RGBA", (w + 40, h + 40), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((8, 14, w + 32, h + 38), 44, fill=(0, 0, 0, 120))
    grad = Image.fromarray(G.vgradient(w, h, G.GOLD_HI, G.GOLD)).convert("RGBA")
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, w - 1, h - 1), 40, fill=255)
    img.paste(grad, (20, 20), mask)
    d.rounded_rectangle((20, 20, w + 19, h + 19), 40, outline=G.GRAPHITE + (255,), width=10)
    d.rounded_rectangle((38, 38, w + 1, h + 1), 26, outline=(120, 74, 8, 255), width=3)
    for text, size, y, name in (("ТЕСТОВЫЙ СЕРВЕР", 40, 58, "bold"), ("24 ЧАСА", 150, 96, "black"), ("БЕСПЛАТНО", 94, 246, "black")):
        f = G.font(name, size)
        d.text((20 + (w - f.getlength(text)) / 2, y), text, font=f, fill=G.GRAPHITE + (255,))
    return G.premult(np.array(img))


def _glow():
    yy, xx = np.mgrid[0:900, 0:900].astype(np.float32)
    r = np.hypot(xx - 450, yy - 450) / 450
    a = np.clip(1 - r, 0, 1) ** 2.2
    out = np.zeros((900, 900, 4), np.float32)
    out[..., :3] = np.array(G.GOLD) * a[..., None] * 0.55
    out[..., 3] = a * 255 * 0.55
    return out.astype(np.uint8)


def shine(sprite, pos):
    """Diagonal light sweep across a premultiplied sprite; pos in 0..1."""
    h, w = sprite.shape[:2]
    xx = np.arange(w)[None, :] + np.arange(h)[:, None] * 0.45
    band = np.exp(-((xx - pos * (w + h * 0.45)) / (w * 0.07)) ** 2)[..., None]
    out = sprite.astype(np.float32)
    out[..., :3] = np.minimum(out[..., 3:4], out[..., :3] + band * out[..., 3:4] * 0.75)
    return out.astype(np.uint8)


def outro(frame, t):
    u = t - CUT["outro"]
    small = cv2.GaussianBlur(cv2.resize(frame, (270, 480), interpolation=cv2.INTER_AREA), (0, 0), 2.2)
    bg = cv2.resize(small, (W_, H_), interpolation=cv2.INTER_LINEAR)
    k = 0.46 - 0.10 * float(smooth(u / 0.5))            # live gameplay stays visible, just dimmed
    cv2.addWeighted(bg, k, np.full_like(bg, G.GRAPHITE), 0.30, 0, dst=frame)
    frame[:] = G.vignette(frame, 0.55)
    pulse = 0.8 + 0.2 * np.sin(2 * np.pi * 0.8 * u)
    G.blit(frame, spr("glow", _glow), 540, 420, scale=1.2, alpha=float(smooth((u - 0.05) / 0.2)) * pulse)
    # logo slam, then a slow "breathing" so the card never sits still
    age = (t - C["logo"]) * FPS
    if age >= 0:
        sc = keyed(age, [0.35, 0.8, 1.1, 1.13, 1.06, 1.02, 1.0]) * (1 + 0.014 * np.sin(2 * np.pi * 0.7 * u))
        logo = spr("logo", _logo)
        for start in (0.28, 1.9):
            p = (u - start) / 0.55
            if 0 <= p <= 1:
                logo = shine(logo, p)
        G.blit(frame, logo, 540, 420, scale=float(sc), angle=1.3 * np.sin(2 * np.pi * 0.45 * u))
    # wordmark wipes in under the mark
    wu = float(smooth((t - C["wordmark"]) / 0.25))
    if wu > 0:
        wm = spr("wordmark", lambda: G.gold_text("leopardix", 144, tracking=-0.015))
        cut = wm.copy()
        cut[:, int(cut.shape[1] * wu):] = 0
        G.blit(frame, cut, 540, 684 + 14 * (1 - wu))
    # offer badge slams in with a shockwave
    age = (t - C["badge"]) * FPS
    if age >= 0:
        ub = t - C["badge"]
        if ub < 0.36:
            ov = frame.copy()
            cv2.circle(ov, (540, 960), int(240 + 620 * ub / 0.36), G.GOLD, 16, cv2.LINE_AA)
            cv2.addWeighted(ov, 0.6 * (1 - ub / 0.36), frame, 1 - 0.6 * (1 - ub / 0.36), 0, dst=frame)
        badge = spr("badge", _badge)
        p = ((ub - 0.35) % 1.5) / 0.5
        if 0 <= p <= 1 and ub > 0.35:
            badge = shine(badge, p)
        sc = keyed(age, SLAM) * (1 + 0.018 * np.sin(2 * np.pi * 1.1 * ub))
        G.blit(frame, badge, 540, 960, scale=float(sc), angle=-4.0, alpha=min(1.0, 0.5 + age * 0.3))
    # slogan lands with the voice
    for key, text, y in (("slogan0", "ЗАХОДИ,", 1238), ("slogan1", "ПОКА НЕ *СДОХЛО!*", 1336)):
        age = (t - C[key] + ED.LEAD) * FPS
        if age >= 0:
            s = spr(key, lambda text=text: G.text_sprite(text, 92, max_width=800))
            G.blit(frame, s, 540, y, scale=keyed(age, POP_STRONG if key == "slogan1" else POP))
    ua = float(smooth((t - C["slogan1"] - 0.55) / 0.25))
    if ua > 0:
        G.blit(frame, spr("url", lambda: G.text_sprite("leopardix.net", 50, name="bold", fill=G.GOLD_HI, stroke=5)), 540, 1424, alpha=ua)


def crt_off(frame, i):
    """Last frames: the picture collapses like a dying CRT ("пока не сдохло")."""
    k = i - (N - 20)
    if k < 0:
        return frame
    out = np.zeros_like(frame)
    if k < 8:
        sy = max(0.004, (1 - (k + 1) / 8) ** 2.2)
        h = max(6, int(H_ * sy))
        band = cv2.resize(frame, (W_, h), interpolation=cv2.INTER_AREA)
        band = cv2.addWeighted(band, 1 - 0.75 * (1 - sy), np.full_like(band, 255), 0.75 * (1 - sy), 0)
        out[H_ // 2 - h // 2:H_ // 2 - h // 2 + h] = band
    elif k < 14:
        w = max(8, int(W_ * (1 - (k - 7) / 6) ** 1.6))
        out[H_ // 2 - 3:H_ // 2 + 3, W_ // 2 - w // 2:W_ // 2 + w // 2] = 255
    elif k < 18:
        v = int(255 * (1 - (k - 13) / 4))
        cv2.circle(out, (W_ // 2, H_ // 2), 7, (v, v, v), -1, cv2.LINE_AA)
    return out


LAYERS = [
    (C["label_de"], CUT["reveal"], tag_germany),
    (C["label_basement"], CUT["igor"], label_basement),
    (C["lag"] - 0.3, CUT["head"], hud_ping),
    (C["trace0"] - 0.15, CUT["orbit"], trace),
    (CUT["spin"] + 0.12, CUT["space"], hud_spin),
    (CUT["space"], CUT["tps"], hud_space),
    (CUT["tps"], CUT["flood_a"], hud_tps),
    (CUT["under"], CUT["support"], bubbles),
    (CUT["flood_a"] + 0.05, CUT["support"], hud_humidity),
    (C["ticket"] - 0.05, CUT["igor2"], ticket),
    (C["igor_ticket"] - 0.05, CUT["trial"], igor_ticket),
    (C["dead"], CUT["outro"], death),
    (CUT["outro"], END + 1, outro),
]


def banner(frame, t):
    """Optional persistent offer pill (off by default: this video reveals the offer in the outro)."""
    def make():
        txt = G.text_sprite("ТЕСТОВЫЙ СЕРВЕР · 24 ЧАСА БЕСПЛАТНО", 34, name="bold", fill=G.GOLD_HI, stroke=4, shadow=False)
        return G.paste(G.panel_sprite(txt.shape[1] + 24, 62, alpha=170, border=(120, 90, 30), radius=30), txt, 12, -8)
    s = spr("banner", make)
    G.blit(frame, s, 540, 178, alpha=0.9)


def captions(frame, t):
    for c in CAPS:
        if c["start"] <= t < c["end"]:
            s = spr(("cap", c["text"]), lambda c=c: G.text_sprite(c["text"], int(118 * c["scale"]), max_width=CAP_W, min_size=76))
            age = (t - c["start"]) * FPS
            G.blit(frame, s, 540, CAP_Y, scale=keyed(age, POP_STRONG if c["strong"] else POP), alpha=min(1.0, 0.55 + age * 0.45))
            return


# ------------------------------------------------------------------ background + main loop

class Background:
    def __init__(self, directory):
        self.dir = directory
        self.shots = json.load(open(os.path.join(directory, "timeline.json")))["shots"]
        self.held = None

    def _open(self, shot, skip=0):
        cmd = ["ffmpeg", "-v", "error", "-i", os.path.join(self.dir, shot["name"] + ".mp4"), "-vf",
               f"select=gte(n\\,{skip}),scale={W_}:{H_}:flags=bicubic", "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        return subprocess.Popen(cmd, stdout=subprocess.PIPE)

    def frames(self, start=0, end=N):
        for shot in self.shots:
            if shot["end"] <= start or shot["start"] >= end:
                continue
            first = max(start, shot["start"])
            p = self._open(shot, first - shot["start"])
            last = None
            try:
                for i in range(first, min(end, shot["end"])):
                    raw = p.stdout.read(W_ * H_ * 3)
                    if len(raw) == W_ * H_ * 3:
                        last = np.frombuffer(raw, np.uint8).reshape(H_, W_, 3).copy()
                    yield i, last.copy()
            finally:
                p.kill()
                p.stdout.close()
                p.wait()


def compose(i, bg, state):
    t = i / FPS
    # freeze-frames: hold + desaturate + slow push
    fz = None
    if C["freeze1"] <= t < CUT["head"]:
        state.setdefault("f1", bg.copy())
        bg, fz = state["f1"].copy(), (t - C["freeze1"]) / (CUT["head"] - C["freeze1"])
    elif C["freeze2"] <= t < CUT["flood_a"]:
        fz = (t - C["freeze2"]) / (CUT["flood_a"] - C["freeze2"])
    in_outro = t >= CUT["outro"]
    zoom = 1.0 + ev(PUNCH, t) + (0.03 + 0.07 * fz if fz is not None else 0.0)
    amp = ev(SHAKE, t) + 9.0 * spin_g(t)
    if in_outro:
        amp *= 0.5
    dx, dy = (np.random.default_rng(i).uniform(-1, 1, 2) * amp) if amp > 0.3 else (0.0, 0.0)
    if zoom != 1.0 or amp > 0.3:
        bg = G.transform(bg, zoom, dx, dy, angle=float(dx) * 0.03)
    if fz is not None:
        bg = G.vignette(G.desaturate(bg, 0.88), 0.6)
    for t0, t1, strength in GLITCH:
        if t0 <= t < t1:
            bg = G.glitch(bg, strength, i)
    frame = np.ascontiguousarray(bg)
    for t0, t1, fn in LAYERS:
        if t0 <= t < t1:
            fn(frame, t)
    captions(frame, t)
    frame = G.flash(frame, min(0.95, ev(FLASH, t)))
    return crt_off(frame, i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bg", default=os.path.join(HERE, "build", "shots"))
    ap.add_argument("--out", default=os.path.join(HERE, "build", "master.mp4"))
    ap.add_argument("--still", nargs="+", help="t1 [t2 ...] out.png : composed frames side by side")
    ap.add_argument("--range", nargs=2, type=float)
    ap.add_argument("--banner", action="store_true", help="keep a subtle offer pill at the top until the outro")
    a = ap.parse_args()
    if a.banner:
        LAYERS.insert(0, (0.0, CUT["outro"], banner))
    bg = Background(a.bg)
    if a.still:
        times, out = [float(x) for x in a.still[:-1]], a.still[-1]
        tiles = []
        for t in times:
            i = min(N - 1, int(round(t * FPS)))
            state = {}
            if C["freeze1"] <= t < CUT["head"]:           # stills inside the freeze need the held frame
                _, f0 = next(bg.frames(int(C["freeze1"] * FPS) + 1, int(C["freeze1"] * FPS) + 2))
                state["f1"] = f0
            _, fr = next(bg.frames(i, i + 1))
            tiles.append(compose(i, fr, state))
        sheet = np.concatenate(tiles, 1)
        if len(tiles) > 1:
            sheet = cv2.resize(sheet, (sheet.shape[1] // 2, sheet.shape[0] // 2), interpolation=cv2.INTER_AREA)
        Image.fromarray(sheet).save(out)
        return
    start, end = (int(a.range[0] * FPS), int(a.range[1] * FPS)) if a.range else (0, N)
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W_}x{H_}", "-r", str(FPS), "-i", "-",
         "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p", "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "8", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv", a.out],
        stdin=subprocess.PIPE)
    state = {}
    for i, fr in bg.frames(start, end):
        enc.stdin.write(compose(i, fr, state).tobytes())
        if i % 300 == 0:
            print(f"frame {i}/{N}", flush=True)
    enc.stdin.close()
    enc.wait()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
