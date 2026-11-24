"""Procedural 16x16 pixel-art textures for the voxel renderer.

Everything here is generated from code (no game assets are used or redistributed).
Each generator returns a list of frames, a frame being a (16, 16, 4) uint8 RGBA array.
"""
import numpy as np

S = 16


def _rng(seed):
    return np.random.default_rng(seed)


def _img(rgb, a=255):
    im = np.zeros((S, S, 4), np.uint8)
    im[..., :3] = rgb
    im[..., 3] = a
    return im


def _clip(a):
    return np.clip(a, 0, 255).astype(np.uint8)


def _jitter(im, amp, seed, levels=4):
    """Quantised per-pixel brightness noise (the chunky look of hand-made pixel art)."""
    r = _rng(seed)
    q = (r.integers(0, levels, (S, S)) - (levels - 1) / 2) / ((levels - 1) / 2) * amp
    out = im.astype(np.float32)
    out[..., :3] += q[..., None]
    out[..., :3] = np.clip(out[..., :3], 0, 255)
    return out.astype(np.uint8)


def _blotch(seed, scale=4):
    """Smooth tileable noise in 0..1."""
    r = _rng(seed)
    g = r.random((S // scale, S // scale))
    g = np.kron(g, np.ones((scale, scale)))
    for _ in range(2):
        g = (g + np.roll(g, 1, 0) + np.roll(g, -1, 0) + np.roll(g, 1, 1) + np.roll(g, -1, 1)) / 5
    g = (g - g.min()) / (g.max() - g.min() + 1e-9)
    return g


def flat(rgb, amp=4, seed=0, border=0):
    im = _jitter(_img(rgb), amp, seed)
    if border:
        f = im.astype(np.float32)
        f[0, :, :3] -= border; f[-1, :, :3] -= border; f[:, 0, :3] -= border; f[:, -1, :3] -= border
        im = _clip(f)
    return [im]


def grass_top(rgb=(104, 168, 62), seed=1):
    return [_jitter(_img(rgb), 16, seed, levels=5)]


def dirt(seed=2):
    im = _jitter(_img((134, 96, 67)), 14, seed, levels=5)
    r = _rng(seed + 100)
    for _ in range(10):
        x, y = r.integers(0, S, 2)
        im[y, x, :3] = (98, 70, 48) if r.random() < 0.6 else (160, 124, 92)
    return [im]


def grass_side(top=(104, 168, 62), seed=3):
    im = dirt(seed)[0]
    g = grass_top(top, seed + 1)[0]
    r = _rng(seed + 2)
    for x in range(S):
        d = 2 + int(r.integers(0, 3))
        im[:d, x] = g[:d, x]
        im[d - 1, x, :3] = _clip(im[d - 1, x, :3].astype(int) - 22)
    return [im]


def stone(base=125, seed=4):
    b = _blotch(seed, 4)
    b = (b + np.roll(b, 2, 1)) / 2
    lv = np.array([base - 20, base - 9, base, base + 10])[np.clip((b * 4).astype(int), 0, 3)]
    im = _img((0, 0, 0))
    im[..., :3] = lv[..., None]
    return [_jitter(im, 5, seed + 1, 3)]


def cobble(base=122, seed=5, moss=False):
    r = _rng(seed)
    pts = r.random((9, 2)) * S
    yy, xx = np.mgrid[0:S, 0:S]
    d = []
    for px, py in pts:
        dx = np.minimum(np.abs(xx - px), S - np.abs(xx - px))
        dy = np.minimum(np.abs(yy - py), S - np.abs(yy - py))
        d.append(np.hypot(dx, dy * 1.25))
    d = np.array(d)
    order = np.argsort(d, 0)
    d1 = np.take_along_axis(d, order[:1], 0)[0]
    d2 = np.take_along_axis(d, order[1:2], 0)[0]
    shade = (base - 22 + (order[0] * 37 % 5) * 11).astype(np.float32)
    edge = (d2 - d1) < 1.05
    v = np.where(edge, base - 52, shade + np.where(d1 < 1.6, 10, 0))
    im = _img((0, 0, 0))
    im[..., :3] = _clip(v)[..., None]
    im = _jitter(im, 5, seed + 1, 3)
    if moss:
        m = _blotch(seed + 9, 4) > 0.58
        im[m, :3] = _clip(np.stack([im[m, 0] * 0.55, im[m, 1] * 0.95 + 14, im[m, 2] * 0.45], -1))
    return [im]


def stone_bricks(seed=6, cracked=False, mossy=False):
    im = _jitter(_img((124, 124, 124)), 6, seed, 4)
    f = im.astype(np.float32)
    for row, seams in ((0, (7,)), (8, (3, 11))):
        f[row, :, :3] += 14
        f[row + 7, :, :3] = 86
        for sx in seams:
            f[row:row + 8, sx, :3] = 86
            f[row:row + 7, (sx + 1) % S, :3] += 10
    im = _clip(f)
    r = _rng(seed + 3)
    if cracked:
        x, y = 4, 1
        for _ in range(11):
            im[y % S, x % S, :3] = 70
            x += int(r.integers(-1, 2)); y += 1
    if mossy:
        m = _blotch(seed + 5, 4) > 0.6
        im[m, :3] = _clip(np.stack([im[m, 0] * 0.6, im[m, 1] * 0.95 + 12, im[m, 2] * 0.5], -1))
    return [im]


def bricks(seed=7):
    im = _jitter(_img((150, 86, 68)), 9, seed, 4)
    for row in range(4):
        y = row * 4 + 3
        im[y, :, :3] = (176, 170, 160)
        off = 0 if row % 2 == 0 else 4
        for sx in (off + 3, off + 11):
            im[row * 4:row * 4 + 3, sx % S, :3] = (176, 170, 160)
    return [im]


def planks(rgb=(162, 130, 78), seed=8):
    im = _img(rgb).astype(np.float32)
    r = _rng(seed)
    for b in range(4):
        tone = r.integers(-10, 11)
        im[b * 4:b * 4 + 4, :, :3] += tone
        im[b * 4 + 3, :, :3] -= 44
        im[b * 4, :, :3] += 8
        j = int(r.integers(0, S))
        im[b * 4:b * 4 + 3, j, :3] -= 36
    im[..., :3] += (_rng(seed + 1).integers(0, 3, (S, S, 1)) - 1) * 5
    return [_clip(im)]


def log_side(seed=9):
    im = _img((104, 82, 50)).astype(np.float32)
    r = _rng(seed)
    for x in range(S):
        if r.random() < 0.45:
            y0, ln = int(r.integers(0, S)), int(r.integers(3, 9))
            for k in range(ln):
                im[(y0 + k) % S, x, :3] = (72, 56, 34)
    im[..., :3] += (_rng(seed + 1).integers(0, 3, (S, S, 1)) - 1) * 6
    return [_clip(im)]


def log_top(seed=10):
    im = _img((104, 82, 50))
    for k, c in enumerate([(184, 148, 92), (156, 122, 72), (184, 148, 92), (156, 122, 72), (184, 148, 92), (150, 116, 68), (176, 140, 86)]):
        im[1 + k:S - 1 - k, 1 + k:S - 1 - k, :3] = c
    return [_jitter(im, 4, seed, 3)]


def leaves(rgb=(58, 122, 36), seed=11):
    im = _jitter(_img(rgb), 22, seed, 5)
    holes = _rng(seed + 1).random((S, S)) < 0.24
    im[holes, 3] = 0
    dark = _rng(seed + 2).random((S, S)) < 0.2
    im[dark, :3] = _clip(im[dark, :3] * 0.7)
    return [im]


def sand(rgb=(219, 207, 163), seed=12):
    return [_jitter(_img(rgb), 8, seed, 5)]


def sandstone(seed=13):
    im = _jitter(_img((216, 203, 155)), 5, seed, 3).astype(np.float32)
    im[0, :, :3] += 12
    im[3, :, :3] -= 22
    im[11, :, :3] -= 16
    im[15, :, :3] -= 26
    return [_clip(im)]


def gravel(seed=14):
    r = _rng(seed)
    pal = np.array([(136, 126, 126), (104, 98, 98), (160, 152, 148), (122, 110, 100), (86, 82, 84)])
    im = _img((0, 0, 0))
    idx = r.integers(0, len(pal), (S // 2, S // 2))
    idx = np.kron(idx, np.ones((2, 2), int))
    idx = np.where(r.random((S, S)) < 0.35, r.integers(0, len(pal), (S, S)), idx)
    im[..., :3] = pal[idx]
    return [im]


def snow(seed=15):
    return [_jitter(_img((238, 248, 250)), 5, seed, 3)]


def ice(seed=16):
    im = _jitter(_img((150, 186, 250)), 6, seed, 3)
    for k in range(0, S, 5):
        for j in range(4):
            im[(k + j) % S, (k * 3 + j) % S, :3] = (205, 225, 255)
    return [im]


def water(frames=4, seed=17):
    out = []
    for f in range(frames):
        im = _img((48, 96, 204)).astype(np.float32)
        b = _blotch(seed, 4)
        b = np.roll(b, f * 2, 1)
        b = np.roll(b, f, 0)
        im[..., :3] += (b[..., None] - 0.5) * 46
        hi = b > 0.78
        im[hi, :3] = (120, 170, 245)
        out.append(_clip(im))
    return out


def lava(frames=4, seed=18):
    out = []
    for f in range(frames):
        b = np.roll(_blotch(seed, 4), f * 2, 0)
        b = np.roll(b, f, 1)
        im = _img((0, 0, 0))
        pal = np.array([(170, 50, 10), (212, 90, 14), (236, 138, 22), (252, 200, 70)])
        im[..., :3] = pal[np.clip((b * 4).astype(int), 0, 3)]
        out.append(im)
    return out


def glow(base=(172, 132, 84), hot=(255, 226, 132), seed=19):
    b = _blotch(seed, 4)
    im = _img(base).astype(np.float32)
    t = np.clip((b - 0.35) * 2.2, 0, 1)[..., None]
    im[..., :3] = im[..., :3] * (1 - t) + np.array(hot) * t
    return [_jitter(_clip(im), 6, seed + 1, 3)]


def neon(rgb, seed=20):
    im = _img(rgb).astype(np.float32)
    im[..., :3] = np.clip(im[..., :3] * 0.72, 0, 255)
    im[2:-2, 2:-2, :3] = rgb
    im[4:-4, 4:-4, :3] = np.clip(np.array(rgb) * 0.55 + 255 * 0.45, 0, 255)
    return [_clip(im)]


def lantern(seed=21):
    im = _img((172, 200, 190)).astype(np.float32)
    im[::4, :, :3] -= 40
    im[:, ::4, :3] -= 40
    im[1:3, 1:3, :3] = 250
    im[9:11, 5:7, :3] = 250
    im[5:7, 13:15, :3] = 250
    im[13:15, 9:11, :3] = 250
    return [_clip(im)]


def iron(seed=22):
    im = _jitter(_img((222, 222, 222)), 3, seed, 3).astype(np.float32)
    im[3::4, :, :3] -= 26
    im[0, :, :3] += 14
    im[:, 0, :3] -= 30; im[:, -1, :3] -= 30; im[-1, :, :3] -= 40
    return [_clip(im)]


def quartz(seed=23):
    im = _jitter(_img((236, 231, 224)), 3, seed, 3).astype(np.float32)
    im[5, 2:12, :3] -= 12
    im[11, 6:15, :3] -= 12
    im[-1, :, :3] -= 16; im[:, -1, :3] -= 10
    return [_clip(im)]


def glass_dark(seed=24):
    yy, xx = np.mgrid[0:S, 0:S]
    t = ((xx + (S - yy)) / (2 * S))[..., None]
    im = _img((0, 0, 0)).astype(np.float32)
    im[..., :3] = np.array((30, 62, 108)) * (1 - t) + np.array((96, 164, 214)) * t
    im[(xx - yy) % 9 == 0, :3] += 36
    im[0, :, :3] = im[-1, :, :3] = (36, 44, 58)
    im[:, 0, :3] = im[:, -1, :3] = (36, 44, 58)
    return [_clip(im)]


def window_lit(seed=25, rgb=(255, 232, 160)):
    im = _img(rgb).astype(np.float32)
    im[..., :3] -= (_rng(seed).integers(0, 3, (S, S, 1))) * 5
    im[0, :, :3] = im[-1, :, :3] = (44, 50, 62)
    im[:, 0, :3] = im[:, -1, :3] = (44, 50, 62)
    im[:, 7:9, :3] = (44, 50, 62)
    return [_clip(im)]


def obsidian(seed=26):
    im = _jitter(_img((22, 18, 34)), 5, seed, 3)
    r = _rng(seed + 1)
    for _ in range(9):
        x, y = r.integers(0, S, 2)
        im[y, x, :3] = (58, 40, 92)
    return [im]


def netherrack(seed=27):
    im = _jitter(_img((112, 54, 52)), 12, seed, 5)
    m = _blotch(seed + 1, 2) > 0.7
    im[m, :3] = (78, 34, 34)
    return [im]


def end_stone(seed=28):
    im = _jitter(_img((220, 222, 160)), 6, seed, 4)
    m = _blotch(seed + 1, 2) > 0.72
    im[m, :3] = (196, 198, 132)
    return [im]


def purpur(seed=29):
    im = _jitter(_img((170, 126, 170)), 5, seed, 3).astype(np.float32)
    im[::8, :, :3] -= 34
    im[:, ::8, :3] -= 34
    im[1::8, :, :3] += 12
    return [_clip(im)]


def blackstone(seed=30):
    return [cobble(60, seed)[0]]


FONT5 = {"T": ["111", "010", "010", "010", "010"], "N": ["1001", "1101", "1011", "1001", "1001"]}


def tnt_side(seed=31):
    im = _jitter(_img((214, 62, 28)), 6, seed, 3).astype(np.float32)
    im[:, 3::4, :3] -= 38
    im[5:11, :, :3] = (236, 236, 230)
    im = _clip(im)
    x = 2
    for ch in "TNT":
        for ry, row in enumerate(FONT5[ch]):
            for rx, c in enumerate(row):
                if c == "1":
                    im[6 + ry - 1, x + rx, :3] = (24, 24, 24)
        x += len(FONT5[ch][0]) + 1
    return [im]


def tnt_top(seed=32):
    im = _jitter(_img((214, 62, 28)), 6, seed, 3)
    im[5:11, 5:11, :3] = (96, 96, 96)
    im[6:10, 6:10, :3] = (40, 40, 40)
    im[7:9, 7:9, :3] = (236, 236, 230)
    return [im]


def concrete_slab(seed=33):
    im = _jitter(_img((150, 150, 146)), 6, seed, 4)
    r = _rng(seed + 1)
    for _ in range(14):
        x, y = r.integers(0, S, 2)
        im[y, x, :3] = (112, 112, 108)
    st = _blotch(seed + 2, 4) > 0.74
    im[st, :3] = _clip(im[st, :3] * 0.86)
    im[:, 0, :3] = (104, 104, 100)
    im[0, :, :3] = (116, 116, 112)
    return [im]


def washer_front(frames=4, seed=34):
    out = []
    yy, xx = np.mgrid[0:S, 0:S]
    cx, cy = 7.5, 9.5
    rr = np.hypot(xx - cx, yy - cy)
    cloth = [(206, 60, 52), (240, 196, 60), (60, 120, 210), (236, 236, 236)]
    for f in range(frames):
        im = _img((232, 234, 236))
        im[0:3, :, :3] = (196, 200, 206)
        im[1, 2:5, :3] = (70, 76, 86)          # display
        im[1, 12, :3] = (230, 52, 44)          # power led
        im[1, 9, :3] = (120, 126, 136)         # knob
        im[3, :, :3] = (168, 172, 180)
        im[:, 0, :3] = im[:, -1, :3] = (176, 180, 188)
        im[-1, :, :3] = (150, 154, 162)
        im[rr < 6.1, :3] = (120, 126, 138)      # door ring
        im[rr < 5.2, :3] = (70, 76, 88)
        im[rr < 4.5, :3] = (26, 40, 62)        # glass
        ang = np.arctan2(yy - cy, xx - cx) + f * np.pi / 2
        for k, c in enumerate(cloth):
            m = (rr < 3.9) & (rr > 0.8) & (np.abs(((ang - k * np.pi / 2 + np.pi) % (2 * np.pi)) - np.pi) < 0.55)
            im[m, :3] = c
        im[(rr < 4.5) & (rr > 3.6) & (xx < cx) & (yy < cy), :3] = (150, 180, 214)   # glass highlight
        out.append(im)
    return out


def washer_side(seed=35):
    im = _jitter(_img((226, 228, 232)), 2, seed, 3)
    im[0:3, :, :3] = (196, 200, 206)
    im[3, :, :3] = (168, 172, 180)
    im[:, 0, :3] = im[:, -1, :3] = (176, 180, 188)
    im[-1, :, :3] = (150, 154, 162)
    im[11:14:2, 4:12, :3] = (150, 154, 162)
    return [im]


def washer_top(seed=36):
    im = _jitter(_img((236, 238, 240)), 2, seed, 3)
    im[0, :, :3] = im[-1, :, :3] = (176, 180, 188)
    im[:, 0, :3] = im[:, -1, :3] = (176, 180, 188)
    im[2:5, 2:8, :3] = (200, 204, 210)
    return [im]


def server_front(frames=4, seed=37, top=False):
    out = []
    r = _rng(seed)
    led_pos = [(row * 4 + 1, x) for row in range(4) for x in (1, 3)]
    states = r.random((frames, len(led_pos)))
    for f in range(frames):
        im = _img((60, 64, 74))
        im[:, 0, :3] = im[:, -1, :3] = (36, 38, 46)
        for row in range(4):
            y = row * 4
            im[y + 3, :, :3] = (28, 30, 36)
            im[y, 1:-1, :3] = (88, 92, 104)
            im[y + 1:y + 3, 6:14:2, :3] = (24, 26, 32)   # vent slots
        if top:
            im[0, :, :3] = (36, 38, 46)
            im[1:3, 5:11, :3] = (255, 190, 50)       # gold badge
        for k, (y, x) in enumerate(led_pos):
            on = states[f, k] > 0.38
            col = (70, 255, 110) if k % 2 == 0 else (255, 176, 40)
            if (top and y < 4):
                continue
            im[y, x, :3] = col if on else (30, 60, 36)
            if on:
                im[y, x, 3] = 200   # marks an emissive texel for the renderer
        out.append(im)
    return out


def server_side(seed=38):
    im = _jitter(_img((56, 60, 70)), 2, seed, 3)
    im[:, 0, :3] = im[:, -1, :3] = (36, 38, 46)
    im[2:14:2, 3:13, :3] = (28, 30, 36)
    return [im]


def server_top(seed=39):
    im = _jitter(_img((64, 68, 78)), 2, seed, 3)
    yy, xx = np.mgrid[0:S, 0:S]
    rr = np.hypot(xx - 7.5, yy - 7.5)
    im[(rr < 5.5) & (rr > 4.5), :3] = (28, 30, 36)
    im[(rr < 4.5) & ((xx + yy) % 2 == 0), :3] = (28, 30, 36)
    return [im]


def shelf_jars(seed=40):
    im = _img((70, 52, 32))
    im[7, :, :3] = im[15, :, :3] = (150, 116, 70)
    im[:, 0, :3] = im[:, -1, :3] = (120, 92, 54)
    fill = [(104, 138, 58), (170, 52, 40), (196, 150, 54), (104, 138, 58), (150, 60, 96), (170, 52, 40)]
    k = 0
    for base in (6, 14):
        for x0 in (2, 7, 11):
            im[base - 4:base + 1, x0:x0 + 3, :3] = fill[k % len(fill)]
            im[base - 4:base + 1, x0, :3] = _clip(np.array(fill[k % len(fill)]) * 1.25 + 20)
            im[base - 5, x0:x0 + 3, :3] = (190, 170, 96)
            k += 1
    return [im]


def crate(seed=41):
    im = planks((150, 114, 66), seed)[0]
    im[0:2, :, :3] = im[-2:, :, :3] = (104, 78, 44)
    im[:, 0:2, :3] = im[:, -2:, :3] = (104, 78, 44)
    for k in range(2, 14):
        im[k, k, :3] = (104, 78, 44)
        im[k, min(k + 1, 13), :3] = (104, 78, 44)
    return [im]


def barrel_side(seed=42):
    im = planks((136, 100, 56), seed)[0]
    im = np.rot90(im).copy()
    im[2:4, :, :3] = (86, 88, 94)
    im[12:14, :, :3] = (86, 88, 94)
    return [im]


def cardboard(seed=43):
    im = _jitter(_img((178, 140, 92)), 5, seed, 3)
    im[:, 7:9, :3] = (206, 186, 140)
    im[0, :, :3] = im[-1, :, :3] = (140, 106, 66)
    im[:, 0, :3] = im[:, -1, :3] = (140, 106, 66)
    return [im]


def pipe(seed=44, vertical=False):
    im = _img((0, 0, 0), 0)
    band = _jitter(_img((150, 86, 44)), 12, seed, 4)
    im[5:11] = band[5:11]
    im[5, :, :3] = (196, 128, 74)
    im[10, :, :3] = (88, 48, 26)
    im[5:11, 3, :3] = (70, 40, 24)
    im[5:11, 12, :3] = (70, 40, 24)
    if vertical:
        im = np.rot90(im).copy()
    return [im]


def cobweb(seed=45):
    im = _img((0, 0, 0), 0)
    c = (178, 178, 174, 255)
    for k in range(S):
        im[k, k] = c
        im[k, S - 1 - k] = c
    im[8, :] = c
    im[:, 8] = c
    for k in (3, 6):
        im[k, k:S - k] = c
        im[S - 1 - k, k:S - k] = c
        im[k:S - k, k] = c
        im[k:S - k, S - 1 - k] = c
    return [im]


def bulb(seed=46):
    im = _img((255, 214, 120))
    im[3:13, 3:13, :3] = (255, 236, 170)
    im[5:11, 5:11, :3] = (255, 252, 226)
    return [im]


def wire(seed=47):
    im = _img((0, 0, 0), 0)
    im[:, 7:9] = (34, 34, 38, 255)
    return [im]


def asphalt(seed=48, line=False):
    im = _jitter(_img((58, 58, 62)), 6, seed, 4)
    if line:
        im[:, 7:9, :3] = (226, 196, 60)
    return [im]


def cactus_side(seed=49):
    im = _jitter(_img((82, 136, 50)), 6, seed, 3).astype(np.float32)
    im[:, 2::5, :3] -= 30
    im[3::6, 4::5, :3] = (224, 228, 190)
    return [_clip(im)]


def nether_bricks(seed=50):
    im = bricks(seed)[0].astype(np.float32)
    g = im[..., :3].mean(-1, keepdims=True)
    im[..., :3] = g * np.array((0.42, 0.20, 0.24))
    return [_clip(im)]


def red_light(seed=51):
    im = _img((255, 40, 30))
    im[4:12, 4:12, :3] = (255, 150, 130)
    return [im]


def antenna(seed=52):
    im = _img((0, 0, 0), 0)
    im[:, 6:10] = (198, 200, 206, 255)
    im[::4, 2:14] = (160, 162, 170, 255)
    return [im]


CONCRETE = {
    "white": (207, 213, 214), "light_gray": (125, 125, 115), "gray": (54, 57, 61), "black": (22, 24, 30),
    "red": (156, 38, 36), "orange": (224, 97, 0), "yellow": (240, 175, 21), "lime": (94, 168, 24),
    "green": (73, 91, 36), "cyan": (21, 119, 136), "light_blue": (35, 137, 198), "blue": (44, 46, 143),
    "purple": (100, 31, 156), "magenta": (169, 48, 159), "pink": (213, 101, 142), "brown": (96, 59, 31),
}


def build():
    """name -> list of frames. Order is stable so indices can be cached."""
    t = {}
    t["grass_top"] = grass_top(); t["grass_side"] = grass_side(); t["dirt"] = dirt()
    t["stone"] = stone(); t["cobble"] = cobble(); t["mossy_cobble"] = cobble(moss=True)
    t["stone_bricks"] = stone_bricks(); t["cracked_bricks"] = stone_bricks(cracked=True); t["mossy_bricks"] = stone_bricks(mossy=True)
    t["bricks"] = bricks(); t["planks"] = planks(); t["log_side"] = log_side(); t["log_top"] = log_top()
    t["leaves"] = leaves(); t["spruce_leaves"] = leaves((40, 86, 50), 61)
    t["sand"] = sand(); t["red_sand"] = sand((190, 104, 44), 62); t["sandstone"] = sandstone(); t["gravel"] = gravel()
    t["snow"] = snow(); t["ice"] = ice(); t["water"] = water(); t["lava"] = lava()
    t["snow_side"] = grass_side((238, 248, 250), 63)
    for name, rgb in CONCRETE.items():
        t["c_" + name] = flat(rgb, 4, 70 + len(t), border=12)
    t["glowstone"] = glow(); t["lantern"] = lantern()
    t["neon_magenta"] = neon((255, 60, 220)); t["neon_cyan"] = neon((40, 235, 255)); t["neon_lime"] = neon((150, 255, 60))
    t["neon_gold"] = neon((255, 190, 40))
    t["iron"] = iron(); t["quartz"] = quartz(); t["glass_dark"] = glass_dark(); t["window_lit"] = window_lit()
    t["obsidian"] = obsidian(); t["netherrack"] = netherrack(); t["end_stone"] = end_stone(); t["purpur"] = purpur()
    t["blackstone"] = blackstone(); t["nether_bricks"] = nether_bricks()
    t["tnt_side"] = tnt_side(); t["tnt_top"] = tnt_top()
    t["concrete_slab"] = concrete_slab()
    t["washer_front"] = washer_front(); t["washer_side"] = washer_side(); t["washer_top"] = washer_top()
    t["server_front"] = server_front(); t["server_front_top"] = server_front(top=True, seed=77)
    t["server_side"] = server_side(); t["server_top"] = server_top()
    t["shelf_jars"] = shelf_jars(); t["crate"] = crate(); t["barrel_side"] = barrel_side(); t["cardboard"] = cardboard()
    t["pipe_h"] = pipe(); t["pipe_v"] = pipe(vertical=True); t["cobweb"] = cobweb(); t["bulb"] = bulb(); t["wire"] = wire()
    t["asphalt"] = asphalt(); t["asphalt_line"] = asphalt(line=True); t["cactus_side"] = cactus_side()
    t["red_light"] = red_light(); t["antenna"] = antenna()
    return t


if __name__ == "__main__":
    # contact sheet for eyeballing the set: python textures.py sheet.png
    import sys

    from PIL import Image

    tex = build()
    names = list(tex)
    cols, cell = 10, 16 * 6 + 6
    sheet = Image.new("RGB", (cols * cell, ((len(names) + cols - 1) // cols) * cell), (40, 40, 60))
    for i, n in enumerate(names):
        im = Image.fromarray(tex[n][0], "RGBA").resize((96, 96), Image.NEAREST)
        sheet.paste(im, ((i % cols) * cell + 3, (i // cols) * cell + 3), im)
    sheet.save(sys.argv[1])
    print(len(names), "textures:", " ".join(names))
