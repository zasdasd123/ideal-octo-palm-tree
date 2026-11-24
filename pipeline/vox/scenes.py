"""World builders for the voxel renderer: parkour themes, the "elite data centre" tower and German's basement."""
import numpy as np
from scipy import interpolate, ndimage

from . import voxlib as V

EYE = 1.62


def fbm(shape, seed, octaves=((48, 1.0), (24, 0.5), (12, 0.25), (6, 0.12))):
    r = np.random.default_rng(seed)
    out = np.zeros(shape)
    for cell, amp in octaves:
        g = r.random((shape[0] // cell + 3, shape[1] // cell + 3))
        out += ndimage.zoom(g, cell, order=3)[:shape[0], :shape[1]] * amp
    out -= out.min()
    return out / out.max()


def fill_terrain(w, h, top, sub, deep, sea=None, sea_block="water"):
    """h[z, x] = number of solid cells in each column."""
    yy = np.arange(w.Y)[:, None, None]
    hh = h[None].astype(int)
    w.v[yy < hh] = w.b(deep)
    w.v[(yy < hh) & (yy >= hh - 4)] = w.b(sub)
    w.v[yy == hh - 1] = w.b(top)
    if sea is not None:
        w.v[(yy < sea) & (yy >= hh)] = w.b(sea_block)


def tree(w, x, y, z, r, leaves="leaves", log="log", tall=False):
    h = int(r.integers(5, 8) if tall else r.integers(4, 7))
    for dy in range(h):
        w.set(x, y + dy, z, log)
    for dy in range(h - 3 if tall else h - 2, h + 2):
        rad = (2 if dy < h else 1) if not tall else max(1, 2 - (dy - (h - 3)) // 2)
        for dx in range(-rad, rad + 1):
            for dz in range(-rad, rad + 1):
                if abs(dx) == rad and abs(dz) == rad and r.random() < 0.6:
                    continue
                if w.get(x + dx, y + dy, z + dz) == 0:
                    w.set(x + dx, y + dy, z + dz, leaves)


# ---------------------------------------------------------------- parkour

def jump_profile(n=60):
    ys, vy, y = [0.0], 0.42, 0.0
    for _ in range(n):
        y += vy
        ys.append(y)
        vy = (vy - 0.08) * 0.98
    return np.array(ys)


_PROFILE = jump_profile()
_SPLINE = interpolate.CubicSpline(np.arange(len(_PROFILE)) / 20.0, _PROFILE)


def airtime(dy):
    k = next(i for i in range(2, len(_PROFILE)) if _PROFILE[i] <= dy and _PROFILE[i] < _PROFILE[i - 1])
    return (k - 1 + (_PROFILE[k - 1] - dy) / (_PROFILE[k - 1] - _PROFILE[k])) / 20.0


THEMES = {
    "plains": dict(look="day", course=("c_lime", "c_yellow", "c_white", "c_light_blue", "c_orange"), y0=74,
                   pillar="stone_bricks", cap="lantern"),
    "desert": dict(look="sunset", course=("c_orange", "c_white", "c_red", "c_yellow", "c_brown"), y0=62,
                   pillar="sandstone", cap="glowstone"),
    "snow": dict(look="snow", course=("c_cyan", "c_white", "c_light_blue", "ice", "c_blue"), y0=84,
                 pillar="quartz", cap="lantern"),
    "night": dict(look="night", course=("neon_magenta", "neon_cyan", "neon_lime", "neon_gold", "c_black"), y0=60,
                  pillar="c_black", cap="neon_cyan"),
    "nether": dict(look="nether", course=("c_red", "blackstone", "glowstone", "nether_bricks", "c_orange"), y0=58,
                   pillar="nether_bricks", cap="glowstone"),
    "end": dict(look="end", course=("purpur", "end_stone", "obsidian", "c_magenta", "neon_magenta"), y0=60,
                pillar="obsidian", cap="neon_magenta"),
}


def _terrain(w, theme, seed):
    r = np.random.default_rng(seed)
    shape = (w.Z, w.X)
    n = fbm(shape, seed)
    if theme == "plains":
        h = (20 + n * 22).astype(int)
        fill_terrain(w, h, "grass", "dirt", "stone", sea=28)
        w.v[(np.arange(w.Y)[:, None, None] == h[None] - 1) & (h[None] <= 29)] = w.b("sand")
        for _ in range(520):
            x, z = int(r.integers(3, w.X - 3)), int(r.integers(3, w.Z - 3))
            if h[z, x] > 29:
                tree(w, x, h[z, x], z, r)
    elif theme == "desert":
        h = (22 + n * 12).astype(int)
        fill_terrain(w, h, "sand", "sand", "sandstone")
        mesa = fbm(shape, seed + 5, ((40, 1.0), (10, 0.3))) > 0.66
        bands = ("c_orange", "c_red", "c_brown", "c_orange", "c_yellow", "c_white", "c_red", "c_orange")
        top = h + (mesa * (14 + 10 * fbm(shape, seed + 6, ((20, 1.0),)))).astype(int)
        yy = np.arange(w.Y)[:, None, None]
        for k, bname in enumerate(bands):
            m = mesa[None] & (yy >= h[None]) & (yy < top[None]) & ((yy // 3) % len(bands) == k)
            w.v[m] = w.b(bname)
        w.v[mesa[None] & (yy == top[None] - 1)] = w.b("red_sand")
        for _ in range(260):
            x, z = int(r.integers(2, w.X - 2)), int(r.integers(2, w.Z - 2))
            if not mesa[z, x]:
                for dy in range(int(r.integers(2, 5))):
                    w.set(x, h[z, x] + dy, z, "cactus")
    elif theme == "snow":
        ridge = 1 - np.abs(fbm(shape, seed + 3, ((40, 1.0), (20, 0.5))) * 2 - 1)
        h = (22 + n * 16 + ridge ** 2 * 34).astype(int)
        fill_terrain(w, h, "snowy_grass", "dirt", "stone", sea=27, sea_block="ice")
        yy = np.arange(w.Y)[:, None, None]
        w.v[(yy < h[None]) & (yy >= h[None] - 3) & (h[None] > 52)] = w.b("snow")
        w.v[(yy < h[None] - 3) & (yy >= h[None] - 9) & (h[None] > 46)] = w.b("stone")
        for _ in range(420):
            x, z = int(r.integers(3, w.X - 3)), int(r.integers(3, w.Z - 3))
            if 27 < h[z, x] < 46:
                tree(w, x, h[z, x], z, r, leaves="spruce_leaves", tall=True)
    elif theme == "night":
        h = (14 + n * 22).astype(int)
        fill_terrain(w, h, "grass", "dirt", "stone", sea=30)
        for _ in range(60):
            x, z = int(r.integers(3, w.X - 3)), int(r.integers(3, w.Z - 3))
            if h[z, x] > 30:
                tree(w, x, h[z, x], z, r)
                w.set(x, h[z, x], z + 1, "lantern")
    elif theme == "nether":
        ridge = 1 - np.abs(fbm(shape, seed + 3, ((32, 1.0), (16, 0.5))) * 2 - 1)
        h = (16 + n * 14 + ridge ** 3 * 26).astype(int)
        fill_terrain(w, h, "netherrack", "netherrack", "netherrack", sea=31, sea_block="lava")
        for _ in range(240):
            x, z = int(r.integers(2, w.X - 2)), int(r.integers(2, w.Z - 2))
            if h[z, x] > 24:
                w.set(x, h[z, x] - 1, z, "glowstone" if r.random() < 0.5 else "blackstone")
    elif theme == "end":
        isl = fbm(shape, seed + 2, ((36, 1.0), (12, 0.35)))
        thick = np.clip((isl - 0.52) * 60, 0, 16).astype(int)
        yy = np.arange(w.Y)[:, None, None]
        top = 36 + (n * 6).astype(int)
        w.v[(yy < top[None]) & (yy >= top[None] - thick[None]) & (thick[None] > 0)] = w.b("end_stone")
        for _ in range(26):
            x, z = int(r.integers(6, w.X - 6)), int(r.integers(6, w.Z - 6))
            if thick[z, x] > 3:
                hh = int(r.integers(10, 30))
                w.box(x - 1, top[z, x], z - 1, x + 1, top[z, x] + hh, z + 1, "obsidian")
                w.set(x, top[z, x] + hh + 1, z, "neon_magenta")
        h = top
    return h


class Run:
    """A simulated first-person sprint-jump run; the course is built under the path it actually takes."""

    def __init__(self, world, theme, seed, length=20.0, fall_at=None, x0=None, z0=22):
        self.w, self.theme, self.fps = world, theme, 240
        r = np.random.default_rng(seed)
        cfg = THEMES[theme]
        x0 = world.X / 2 if x0 is None else x0
        dt = 1.0 / self.fps
        x, z, gy = float(x0), float(z0), float(cfg["y0"])
        heading, target = 0.0, 0.0
        state, run_left, t_air, T, dy = "run", 2.6, 0.0, 0.0, 0
        ts, xs, ys, zs, gys, air, plat = [], [], [], [], [], [], []
        pid, t = 0, 0.0
        self.fall_t = None
        falling = False
        while t < length:
            d = target - heading
            heading += np.clip(d, -3.0 * dt, 3.0 * dt)
            v = 7.0 if state == "air" else 5.8
            x += np.sin(heading) * v * dt
            z += np.cos(heading) * v * dt
            if state == "run":
                y = gy
                run_left -= v * dt
                if run_left <= 0:
                    state, t_air = "air", 0.0
                    if fall_at is not None and self.fall_t is None and t >= fall_at:
                        falling, self.fall_t, T, dy = True, t, 1e9, 0
                        self.miss = (x + np.sin(heading) * 6.6, gy, z + np.cos(heading) * 6.6, heading)
                    else:
                        lo, hi = cfg["y0"] - 5, cfg["y0"] + 7
                        dy = int(r.choice([1, 0, 0, 0, -1, -1, -2] if gy > lo + 2 else [1, 1, 0]))
                        if gy + dy > hi:
                            dy = -1
                        T = airtime(dy)
                        target = float(np.clip(-0.05 * (x - x0) + r.normal(0, 0.30), -0.62, 0.62))
            else:
                t_air += dt
                y = gy + float(_SPLINE(min(t_air, 2.9))) if t_air < 2.9 else gy + _PROFILE[58] - (t_air - 2.9) * 62
                if not falling and t_air >= T:
                    gy += dy
                    y = gy
                    state, run_left = "run", float(r.uniform(1.0, 2.5))
                    pid += 1
            ts.append(t); xs.append(x); ys.append(y); zs.append(z); gys.append(gy)
            air.append(state == "air"); plat.append(pid)
            t += dt
        self.t = np.array(ts)
        self.p = np.stack([xs, ys, zs], 1)
        self.gy, self.air, self.plat = np.array(gys), np.array(air), np.array(plat)
        self._build(cfg, r)
        self._camera()

    def _build(self, cfg, r):
        w = self.w
        ground = ~self.air
        for i in np.where(ground)[0][::4]:
            x, y, z = self.p[i]
            name = cfg["course"][self.plat[i] % len(cfg["course"])]
            cx, cz, cy = int(np.floor(x)), int(np.floor(z)), int(round(self.gy[i])) - 1
            w.set(cx, cy, cz, name)
            fx, fz = x - cx, z - cz
            if fx < 0.3: w.set(cx - 1, cy, cz, name)
            if fx > 0.7: w.set(cx + 1, cy, cz, name)
            if fz < 0.3: w.set(cx, cy, cz - 1, name)
            if fz > 0.7: w.set(cx, cy, cz + 1, name)
        if self.fall_t is not None:
            mx, my, mz, hd = self.miss
            for k in range(3):
                w.set(int(mx + np.sin(hd) * k), int(my) - 1, int(mz + np.cos(hd) * k), cfg["course"][0])
        # landmark pillars beside the course for parallax
        for zc in range(30, w.Z - 10, 17):
            i = int(np.argmin(np.abs(self.p[:, 2] - zc)))
            side = 1 if (zc // 17) % 2 else -1
            px = int(self.p[i, 0] + side * r.integers(6, 12))
            top = int(self.gy[i] + r.integers(-4, 6))
            if np.min(np.hypot(self.p[:, 0] - px, self.p[:, 2] - zc)) < 4.5:
                continue
            w.box(px, 0, zc, px + 1, top, zc + 1, cfg["pillar"])
            w.box(px - 1, top + 1, zc - 1, px + 2, top + 1, zc + 2, cfg["pillar"])
            w.box(px, top + 2, zc, px + 1, top + 2, zc + 1, cfg["cap"])

    def _camera(self):
        t, p = self.t, self.p
        la = np.clip(np.searchsorted(t, t + 0.55), 0, len(t) - 1)
        tgt = np.stack([p[la, 0], self.gy[la] + 0.2, p[la, 2]], 1)
        eye = p + np.array([0, EYE, 0])
        if self.fall_t is not None:
            f = np.clip((t - self.fall_t - 0.35) / 0.5, 0, 1)[:, None]
            tgt = tgt * (1 - f) + (eye + np.array([0.0, -6.0, 2.5])) * f
        yaw, pitch = V.look_at(eye, tgt)
        pitch = np.clip(pitch, np.radians(-75 if self.fall_t is not None else -34), np.radians(-5))
        k = 1 - np.exp(-1.0 / (0.11 * self.fps))
        for arr in (yaw, pitch):
            for i in range(1, len(arr)):
                arr[i] = arr[i - 1] + (arr[i] - arr[i - 1]) * k
        # footstep bob on the ground, tiny dip after landing
        dist = np.concatenate([[0], np.cumsum(np.hypot(np.diff(p[:, 0]), np.diff(p[:, 2])))])
        bob = np.where(self.air, 0.0, np.abs(np.sin(dist * 2.4)) * 0.035)
        self.eye = eye + np.stack([np.zeros_like(bob), bob, np.zeros_like(bob)], 1)
        self.yaw, self.pitch = yaw, pitch

    def sample(self, times):
        """Camera at arbitrary run times (seconds) -> pos (n,3), yaw, pitch."""
        times = np.clip(times, 0, self.t[-1])
        pos = np.stack([np.interp(times, self.t, self.eye[:, k]) for k in range(3)], 1)
        return pos, np.interp(times, self.t, self.yaw), np.interp(times, self.t, self.pitch)


def parkour(pal, theme, seed, length=20.0, fall_at=None):
    w = V.World(pal, 128, 128, 288)
    _terrain(w, theme, seed)
    run = Run(w, theme, seed + 1, length=length, fall_at=fall_at)
    return w, run


# ---------------------------------------------------------------- tower

def tower(pal, seed=7):
    w = V.World(pal, 128, 144, 176)
    r = np.random.default_rng(seed)
    n = fbm((w.Z, w.X), seed)
    h = (18 + n * 10).astype(int)
    zz, xx = np.mgrid[0:w.Z, 0:w.X]
    d = np.hypot(xx - 64, zz - 118)
    h = np.where(d < 30, 22, np.where(d < 44, (22 + (h - 22) * (d - 30) / 14).astype(int), h))
    fill_terrain(w, h, "grass", "dirt", "stone", sea=20)
    for _ in range(420):
        x, z = int(r.integers(3, w.X - 3)), int(r.integers(3, w.Z - 3))
        if np.hypot(x - 64, z - 118) > 26 and h[z, x] > 21 and abs(x - 64) > 6:
            tree(w, x, h[z, x], z, r)
    g = 22                                    # plaza level (first free cell)
    w.box(46, g - 1, 100, 82, g - 1, 136, "c_light_gray")
    w.box(61, g - 1, 0, 67, g - 1, 100, "asphalt")
    w.box(64, g - 1, 0, 64, g - 1, 100, "asphalt_line")
    x0, x1, z0, z1, top = 56, 72, 110, 126, g + 84
    w.box(x0, g, z0, x1, top, z1, "glass_dark")
    for y in range(g, top + 1):
        if (y - g) % 4 == 0:
            w.box(x0, y, z0, x1, y, z1, "quartz")
    for x in (x0, x0 + 4, x0 + 8, x0 + 12, x1):
        w.box(x, g, z0, x, top, z0, "c_white")
        w.box(x, g, z1, x, top, z1, "c_white")
    for z in (z0, z0 + 4, z0 + 8, z0 + 12, z1):
        w.box(x0, g, z, x0, top, z, "c_white")
        w.box(x1, g, z, x1, top, z, "c_white")
    for y in range(g + 1, top):
        if (y - g) % 4 == 0:
            continue
        for x in range(x0 + 1, x1):
            if (x - x0) % 4 and r.random() < 0.22:
                w.set(x, y, z0, "window_lit")
        for z in range(z0 + 1, z1):
            if (z - z0) % 4 and r.random() < 0.22:
                w.set(x0, y, z, "window_lit")
                w.set(x1, y, z, "window_lit")
    w.box(x0, top - 3, z0, x1, top - 3, z1, "neon_gold")
    w.box(x0 + 3, top + 1, z0 + 3, x1 - 3, top + 3, z1 - 3, "c_gray")
    w.box(64, top + 4, 118, 64, top + 16, 118, "antenna")
    w.set(64, top + 17, 118, "red_light")
    w.box(59, top + 4, 113, 60, top + 5, 114, "iron")
    w.box(62, g, z0 - 1, 66, g + 3, z0 - 1, "c_black")       # entrance
    w.box(63, g, z0 - 1, 65, g + 2, z0 - 1, "window_lit")
    for x, z in ((52, 104), (76, 104), (52, 132), (76, 132), (58, 92), (70, 92), (58, 76), (70, 76)):
        w.box(x, g, z, x, g + 3, z, "iron")
        w.set(x, g + 4, z, "lantern")
    return w


# ---------------------------------------------------------------- basement

BASEMENT = dict(x0=10, x1=20, z0=8, z1=26, floor=4, ceil=8, stack=(15.5, 4.0, 24.5), lamp=(13, 7, 20))


def basement(pal, seed=3):
    """German's basement. Interior cells x 10..20, z 8..26, y 4..7; the camera looks towards +z."""
    b = BASEMENT
    w = V.World(pal, 32, 24, 40)
    r = np.random.default_rng(seed)
    w.v[:12] = w.b("stone")
    w.v[12:15] = w.b("dirt")
    w.v[15] = w.b("grass")
    w.box(b["x0"], b["floor"], b["z0"], b["x1"], b["ceil"] - 1, b["z1"], "air")
    wall = ["cobble"] * 11 + ["mossy_cobble"] * 4 + ["stone_bricks"] * 3 + ["cracked_bricks", "mossy_bricks", "bricks"]
    for y in range(b["floor"], b["ceil"]):
        for z in range(b["z0"] - 1, b["z1"] + 2):
            w.set(b["x0"] - 1, y, z, wall[int(r.integers(len(wall)))])
            w.set(b["x1"] + 1, y, z, wall[int(r.integers(len(wall)))])
        for x in range(b["x0"] - 1, b["x1"] + 2):
            w.set(x, y, b["z0"] - 1, wall[int(r.integers(len(wall)))])
            w.set(x, y, b["z1"] + 1, wall[int(r.integers(len(wall)))])
    floor = ["stone"] * 6 + ["cobble"] * 3 + ["gravel"] * 2 + ["dirt"]
    for z in range(b["z0"], b["z1"] + 1):
        for x in range(b["x0"], b["x1"] + 1):
            w.set(x, b["floor"] - 1, z, floor[int(r.integers(len(floor)))])
            w.set(x, b["ceil"], z, "concrete_slab")
    f, c = b["floor"], b["ceil"] - 1
    # window well in the right wall: cold daylight against the warm bulb
    for z in (15, 16):
        w.set(b["x1"] + 1, c, z, "air")
        w.box(b["x1"] + 2, c - 1, z, b["x1"] + 4, 23, z, "air")
    w.box(b["x1"] + 2, c - 2, 14, b["x1"] + 4, c - 2, 17, "gravel")
    w.set(*b["lamp"], "bulb")
    for lx, lz in ((19, 20), (14, 12), (18, 11)):
        w.set(lx, c, lz, "bulb")
    # back wall: shelves of pickle jars on the left, pipes, clutter on the right
    for x in (11, 12, 13):
        for y in (f, f + 1, f + 2):
            w.set(x, y, 26, "shelf")
    for x in range(10, 21):
        w.set(x, c, 26, "pipe_h")
    for y in range(f, c):
        w.set(19, y, 26, "pipe_v")
    w.set(18, f, 25, "barrel"); w.set(20, f, 25, "crate"); w.set(20, f + 1, 25, "cardboard")
    w.set(17, f, 24, "tnt"); w.set(17, f, 26, "crate"); w.set(17, f + 1, 26, "cardboard"); w.set(14, f, 25, "barrel")
    w.set(10, c, 26, "cobweb"); w.set(20, c, 9, "cobweb"); w.set(10, c, 9, "cobweb")
    w.set(10, f, 17, "crate"); w.set(10, f + 1, 17, "cardboard"); w.set(10, f, 18, "barrel"); w.set(11, f, 17, "cardboard")
    w.set(20, f, 12, "crate"); w.set(20, f, 13, "crate"); w.set(20, f + 1, 13, "barrel"); w.set(19, f, 13, "cardboard")
    # puddle under the leaking pipe
    for x, z in ((16, 21), (17, 21), (16, 22), (17, 22), (18, 22)):
        w.set(x, f - 1, z, "water")
    w.add_object([[["washer"]]])
    w.add_object([[["server_lo"]], [["server_hi"]]])
    return w
