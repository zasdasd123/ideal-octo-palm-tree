"""2D helpers for the compositor: text sprites, stamps, blits and frame effects (Pillow + OpenCV).

Frames are HxWx3 uint8 RGB. Sprites are HxWx4 uint8 with premultiplied alpha.
"""
import os
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FONTS = dict(black="Montserrat-Black.ttf", bold="Montserrat-ExtraBold.ttf", russo="RussoOne-Regular.ttf",
             mc="Monocraft.ttf", mono="JetBrainsMono-ExtraBold.ttf")
WHITE, BLACK, YELLOW = (255, 255, 255), (0, 0, 0), (255, 226, 0)
GOLD, GOLD_HI, GOLD_LO, GRAPHITE = (245, 166, 35), (255, 214, 102), (196, 120, 18), (24, 27, 24)
RED, GREEN, CYAN = (232, 44, 44), (52, 208, 96), (70, 208, 255)
FUNCTION_WORDS = {"В", "НА", "ЗА", "У", "С", "К", "О", "ПО", "ИЗ", "ОТ", "ДО", "БЕЗ", "ПОД", "НАД", "ПРИ", "ПРО", "ДЛЯ",
                  "ЧЕРЕЗ", "И", "А", "НО", "КАК", "НЕ"}


@lru_cache(maxsize=None)
def font(name, size):
    return ImageFont.truetype(os.path.join(ROOT, "assets", "fonts", FONTS[name]), int(size))


def premult(rgba):
    out = np.asarray(rgba, np.uint8).copy()
    out[..., :3] = (out[..., :3].astype(np.uint16) * out[..., 3:4] // 255).astype(np.uint8)
    return out


def segments(text):
    """'ЗА *5 ЗВЁЗД!*' -> [('ЗА ', False), ('5 ЗВЁЗД!', True)]"""
    return [(s, i % 2 == 1) for i, s in enumerate(text.split("*")) if s]


def _lines(text, fnt, max_width):
    """Split marked-up text into at most two lines; a line never ends on a preposition."""
    plain = text.replace("*", "")
    if max_width is None or fnt.getlength(plain) <= max_width or " " not in plain:
        return [text]
    words = text.split(" ")
    best = None
    for k in range(1, len(words)):
        a, b = " ".join(words[:k]), " ".join(words[k:])
        if a.count("*") % 2:                      # keep the highlight span balanced on both lines
            a, b = a + "*", "*" + b
        last = a.replace("*", "").strip(" —,.!?…:«»").split()[-1]
        if last in FUNCTION_WORDS:
            continue
        wmax = max(fnt.getlength(a.replace("*", "")), fnt.getlength(b.replace("*", "")))
        if best is None or wmax < best[0]:
            best = (wmax, [a, b])
    return best[1] if best else [text]


def text_sprite(text, size, name="black", fill=WHITE, hi=YELLOW, stroke=None, stroke_fill=BLACK, shadow=True,
                max_width=None, min_size=None, wrap=True, spacing=1.06):
    """Bold caption text with a thick outline and soft drop shadow. *spans* use the highlight colour."""
    plain = text.replace("*", "")
    fnt = font(name, size)
    if max_width and fnt.getlength(plain) > max_width:
        fit = int(size * max_width / fnt.getlength(plain))
        if min_size is None or fit >= min_size or not wrap or " " not in plain:
            size = max(fit, 8)
        else:
            size = min_size
        fnt = font(name, size)
    lines = _lines(text, fnt, max_width) if wrap else [text]
    if max_width:
        widest = max(fnt.getlength(ln.replace("*", "")) for ln in lines)
        if widest > max_width:
            size = int(size * max_width / widest)
            fnt = font(name, size)
    stroke = int(round(size * 0.09)) if stroke is None else stroke
    asc, desc = fnt.getmetrics()
    lh = int((asc + desc) * spacing)
    widths = [fnt.getlength(ln.replace("*", "")) for ln in lines]
    pad = stroke + 22
    Wd, Hd = int(max(widths)) + 2 * pad, lh * len(lines) + 2 * pad
    img = Image.new("RGBA", (Wd, Hd), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for outline in (True, False):
        for li, ln in enumerate(lines):
            x = (Wd - widths[li]) / 2
            y = pad + li * lh
            for seg, on in segments(ln):
                if outline and stroke:
                    d.text((x, y), seg, font=fnt, fill=stroke_fill + (255,), stroke_width=stroke, stroke_fill=stroke_fill + (255,))
                elif not outline:
                    d.text((x, y), seg, font=fnt, fill=(hi if on else fill) + (255,))
                x += fnt.getlength(seg)
    if shadow:
        a = img.split()[3]
        sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
        sh.paste((0, 0, 0, 170), (0, int(size * 0.07)), a)
        sh = sh.filter(ImageFilter.GaussianBlur(max(3, size * 0.05)))
        img = Image.alpha_composite(sh, img)
    return premult(np.array(img))


def mc_text(text, size, fill=WHITE, name="mc"):
    """Game-HUD style text: pixel font with the hard offset shadow."""
    fnt = font(name, size)
    off = max(2, int(size / 9))
    w, (asc, desc) = int(fnt.getlength(text)) + off + 8, fnt.getmetrics()
    img = Image.new("RGBA", (w, asc + desc + off + 8), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((4 + off, 4 + off), text, font=fnt, fill=tuple(int(c * 0.25) for c in fill) + (255,))
    d.text((4, 4), text, font=fnt, fill=fill + (255,))
    return premult(np.array(img))


def _distress(img, seed, amount=0.16):
    r = np.random.default_rng(seed)
    a = np.array(img)
    h, w = a.shape[:2]
    n = cv2.resize(r.random((h // 6 + 2, w // 6 + 2)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    n2 = r.random((h, w)).astype(np.float32)
    keep = (n > amount * 1.6) & (n2 > amount * 0.5)
    a[..., 3] = (a[..., 3] * np.where(keep, 1.0, 0.25)).astype(np.uint8)
    return Image.fromarray(a)


def stamp_sprite(lines, color, size, seed=1, border=None, sub=None):
    """Rubber-stamp: bordered box + heavy text in one ink colour, slightly worn, on a dark backing."""
    fnt = font("black", size)
    border = border or max(8, size // 9)
    widths = [fnt.getlength(ln) for ln in lines]
    asc, desc = fnt.getmetrics()
    lh = int((asc + desc) * 0.98)
    sfnt = font("black", int(size * 0.5)) if sub else None
    sh = int(size * 0.70) if sub else 0
    padx, pady = int(size * 0.42), int(size * 0.20)
    w, h = int(max(widths)) + 2 * (padx + border), lh * len(lines) + sh + 2 * (pady + border)
    m = 16
    img = Image.new("RGBA", (w + 2 * m, h + 2 * m), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    rad = int(size * 0.22)
    d.rounded_rectangle((m - 5, m - 5, m + w + 5, m + h + 5), rad + 5, fill=(0, 0, 0, 150), outline=(0, 0, 0, 255), width=6)
    ink = Image.new("RGBA", img.size, (0, 0, 0, 0))
    di = ImageDraw.Draw(ink)
    di.rounded_rectangle((m, m, m + w, m + h), rad, outline=color + (255,), width=border)
    for li, ln in enumerate(lines):
        di.text((m + (w - widths[li]) / 2, m + border + pady + li * lh), ln, font=fnt, fill=color + (255,))
    if sub:
        di.text((m + (w - sfnt.getlength(sub)) / 2, m + border + pady + lh * len(lines)), sub, font=sfnt, fill=color + (255,))
    ink = _distress(ink, seed)
    return premult(np.array(Image.alpha_composite(img, ink)))


def hazard_sprite(lines, size, fill=(255, 212, 0)):
    """Black-on-yellow warning plate with hazard stripes."""
    fnt = font("black", size)
    widths = [fnt.getlength(ln) for ln in lines]
    asc, desc = fnt.getmetrics()
    lh = int((asc + desc) * 0.98)
    padx, stripe = int(size * 0.5), int(size * 0.26)
    w, h = int(max(widths)) + 2 * padx, lh * len(lines) + 2 * stripe + int(size * 0.24)
    m = 14
    img = Image.new("RGBA", (w + 2 * m, h + 2 * m), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle((m - 8, m - 8, m + w + 8, m + h + 8), fill=(0, 0, 0, 255))
    d.rectangle((m, m, m + w, m + h), fill=fill + (255,))
    for y0 in (m, m + h - stripe):
        for x in range(-stripe, w + stripe, stripe * 2):
            d.polygon([(m + x, y0 + stripe), (m + x + stripe, y0), (m + x + 2 * stripe, y0), (m + x + stripe, y0 + stripe)], fill=(0, 0, 0, 255))
    d.rectangle((0, 0, m - 8, img.size[1]), fill=(0, 0, 0, 0))
    d.rectangle((m + w + 8, 0, img.size[0], img.size[1]), fill=(0, 0, 0, 0))
    for li, ln in enumerate(lines):
        d.text((m + (w - widths[li]) / 2, m + stripe + int(size * 0.10) + li * lh), ln, font=fnt, fill=(0, 0, 0, 255))
    return premult(np.array(img))


def panel_sprite(w, h, fill=(16, 18, 22), alpha=214, border=(84, 92, 108), radius=16, bar=None):
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, w - 1, h - 1), radius, fill=fill + (alpha,), outline=border + (255,), width=3)
    if bar:
        d.rounded_rectangle((0, 0, 14, h - 1), 6, fill=bar + (255,))
    return premult(np.array(img))


def star_sprite(size, fill=(255, 196, 48), edge=(40, 28, 0)):
    s = size * 4
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c, pts = s / 2, []
    for k in range(10):
        r = s * (0.47 if k % 2 == 0 else 0.20)
        a = -np.pi / 2 + k * np.pi / 5
        pts.append((c + r * np.cos(a), c + r * np.sin(a)))
    d.polygon(pts, fill=fill + (255,), outline=edge + (255,), width=s // 22)
    return premult(np.array(img.resize((size, size), Image.LANCZOS)))


def paste(canvas, spr, x, y):
    """'Over' a premultiplied sprite onto a premultiplied RGBA canvas at top-left (x, y), clipped."""
    h, w = spr.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(canvas.shape[1], x + w), min(canvas.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return canvas
    s = spr[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.uint16)
    reg = canvas[y0:y1, x0:x1].astype(np.uint16)
    canvas[y0:y1, x0:x1] = (s + reg * (255 - s[..., 3:4]) // 255).clip(0, 255).astype(np.uint8)
    return canvas


def blit(dst, spr, cx, cy, scale=1.0, angle=0.0, alpha=1.0):
    """Alpha-blend a premultiplied sprite centred at (cx, cy)."""
    if alpha <= 0.003 or scale <= 0.01:
        return
    if scale != 1.0 or angle != 0.0:
        h, w = spr.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        c, s = abs(M[0, 0]), abs(M[0, 1])
        nw, nh = int(w * c + h * s) + 2, int(w * s + h * c) + 2
        M[0, 2] += nw / 2 - w / 2
        M[1, 2] += nh / 2 - h / 2
        spr = cv2.warpAffine(spr, M, (nw, nh), flags=cv2.INTER_AREA if scale < 0.6 else cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    h, w = spr.shape[:2]
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    a0, b0 = max(0, -x0), max(0, -y0)
    a1, b1 = min(w, dst.shape[1] - x0), min(h, dst.shape[0] - y0)
    if a1 <= a0 or b1 <= b0:
        return
    s = spr[b0:b1, a0:a1].astype(np.uint16)
    roi = dst[y0 + b0:y0 + b1, x0 + a0:x0 + a1]
    al = int(alpha * 256)
    inv = 255 - (s[..., 3:4] * al >> 8)
    roi[:] = ((s[..., :3] * al >> 8) + (roi.astype(np.uint16) * inv + 127) // 255).clip(0, 255).astype(np.uint8)


def blit_tl(dst, spr, x, y, alpha=1.0):
    blit(dst, spr, x + spr.shape[1] / 2, y + spr.shape[0] / 2, alpha=alpha)


def transform(frame, scale=1.0, dx=0.0, dy=0.0, angle=0.0):
    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += dx
    M[1, 2] += dy
    return cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def desaturate(frame, amount):
    g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return cv2.addWeighted(frame, 1 - amount, cv2.cvtColor(g, cv2.COLOR_GRAY2RGB), amount, 0)


def flash(frame, a, color=WHITE):
    if a <= 0.004:
        return frame
    return cv2.addWeighted(frame, 1 - a, np.full_like(frame, color), a, 0)


def dim(frame, k):
    return cv2.convertScaleAbs(frame, alpha=k, beta=0)


def glitch(frame, strength, seed):
    """Digital tear: RGB split + displaced bands + a few dropped scanlines."""
    r = np.random.default_rng(seed)
    out = frame.copy()
    k = int(6 + 26 * strength)
    out[..., 0] = np.roll(frame[..., 0], -k, axis=1)
    out[..., 2] = np.roll(frame[..., 2], k, axis=1)
    h = frame.shape[0]
    for _ in range(int(3 + 9 * strength)):
        y0 = int(r.integers(0, h - 40))
        bh = int(r.integers(12, 130))
        out[y0:y0 + bh] = np.roll(out[y0:y0 + bh], int(r.integers(-150, 150) * strength), axis=1)
    for _ in range(int(4 * strength) + 1):
        y0 = int(r.integers(0, h - 8))
        out[y0:y0 + int(r.integers(2, 7))] = (out[y0:y0 + 1] * 0.25).astype(np.uint8)
    return out


@lru_cache(maxsize=4)
def _vignette(h, w, strength):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2)) / 1.414
    return (1 - strength * np.clip(d, 0, 1) ** 2.2)[..., None].astype(np.float32)


def vignette(frame, strength=0.5):
    return (frame * _vignette(frame.shape[0], frame.shape[1], round(strength, 2))).astype(np.uint8)


def vgradient(w, h, top, bottom):
    t = np.linspace(0, 1, h)[:, None, None]
    return np.broadcast_to((np.array(top) * (1 - t) + np.array(bottom) * t), (h, w, 3)).astype(np.uint8)


def gold_text(text, size, name="russo", stroke=8, tracking=0.0):
    """Wordmark in the brand's gold-on-graphite."""
    fnt = font(name, size)
    adv = [fnt.getlength(ch) + tracking * size for ch in text]
    asc, desc = fnt.getmetrics()
    pad = stroke + 20
    w, h = int(sum(adv)) + 2 * pad, asc + desc + 2 * pad
    mask = Image.new("L", (w, h), 0)
    edge = Image.new("L", (w, h), 0)
    dm, de = ImageDraw.Draw(mask), ImageDraw.Draw(edge)
    x = pad
    for ch, a in zip(text, adv):
        de.text((x, pad), ch, font=fnt, fill=255, stroke_width=stroke, stroke_fill=255)
        dm.text((x, pad), ch, font=fnt, fill=255)
        x += a
    grad = vgradient(w, h, GOLD_HI, GOLD_LO).copy()
    grad[: pad + int(asc * 0.52)] = np.maximum(grad[: pad + int(asc * 0.52)], np.array(GOLD, np.uint8))
    out = np.zeros((h, w, 4), np.uint8)
    e, m = np.array(edge)[..., None] / 255.0, np.array(mask)[..., None] / 255.0
    rgb = np.array(GRAPHITE) * e * (1 - m) + grad * m
    out[..., :3] = rgb.astype(np.uint8)
    out[..., 3] = (np.maximum(e, m)[..., 0] * 255).astype(np.uint8)
    sh = Image.fromarray(out[..., 3]).filter(ImageFilter.GaussianBlur(10))
    shadow = np.zeros_like(out)
    shadow[..., 3] = (np.array(sh) * 0.7).astype(np.uint8)
    res = Image.alpha_composite(Image.fromarray(shadow), Image.fromarray(out))
    return premult(np.array(res))
