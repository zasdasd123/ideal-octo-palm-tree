"""Scene building + render driver for the voxel raycaster (vox.c)."""
import os
import struct
import subprocess

import numpy as np

from . import textures

NF = 112
P = dict(CAM=0, YAW=3, PITCH=4, ROLL=5, FOV=6, TIME=7, ZEN=8, HOR=11, FOG=14, FOG0=17, FOG1=18, SUN=19, SUNCOL=22,
         SUNSIZE=25, SKYL=26, BLKL=29, AMB=32, EXPO=33, GAMMA=34, CLOUDY=35, CLOUDA=36, STARS=37, FLOODY=38, FLOODR=39,
         WTINT=43, WDENS=46, LAMP=47, SPEC=50, NOBJ=52, OBJ=56)
OBJ_NF = 13
SOLID, CUTOUT, WATER, BRIGHT = 1, 2, 4, 8
FACES = ("+x", "-x", "+y", "-y", "+z", "-z")
HERE = os.path.dirname(os.path.abspath(__file__))
VOX_BIN = os.environ.get("VOX_BIN", os.path.join(os.path.expanduser("~"), ".cache", "leopardix-video", "vox"))


def build_binary():
    os.makedirs(os.path.dirname(VOX_BIN), exist_ok=True)
    src = os.path.join(HERE, "vox.c")
    if not os.path.exists(VOX_BIN) or os.path.getmtime(VOX_BIN) < os.path.getmtime(src):
        subprocess.run(["gcc", "-O3", "-march=native", "-ffast-math", "-fopenmp", "-o", VOX_BIN, src, "-lm"], check=True)
    return VOX_BIN


class Palette:
    """Texture atlas + block table shared by all scenes."""

    def __init__(self):
        self.tex = textures.build()
        self.tex_index, self.frames = {}, []
        for name, fr in self.tex.items():
            self.tex_index[name] = len(self.frames)
            self.frames.extend(fr)
        self.blocks = [dict(name="air", tex=[0] * 6, flags=0, emit=0, frames=1, ticks=1, mask=0)]
        self.id = {"air": 0}
        self._define()

    def add(self, name, all=None, top=None, bottom=None, side=None, front=None, flags=SOLID, emit=0, ticks=4, anim=None):
        side = side or all
        faces = [side, side, top or all or side, bottom or top or all or side, side, front or side]
        frames = max(len(self.tex[f]) for f in faces)
        mask = sum(1 << i for i, f in enumerate(faces) if len(self.tex[f]) > 1)
        self.id[name] = len(self.blocks)
        self.blocks.append(dict(name=name, tex=[self.tex_index[f] for f in faces], flags=flags, emit=emit,
                                frames=frames, ticks=ticks, mask=mask))
        return self.id[name]

    def _define(self):
        a = self.add
        a("grass", top="grass_top", bottom="dirt", side="grass_side")
        for n in ("dirt", "stone", "cobble", "mossy_cobble", "stone_bricks", "cracked_bricks", "mossy_bricks", "bricks",
                  "planks", "sand", "red_sand", "sandstone", "gravel", "snow", "ice", "iron", "quartz", "glass_dark",
                  "obsidian", "netherrack", "end_stone", "purpur", "blackstone", "nether_bricks", "concrete_slab",
                  "crate", "cardboard", "asphalt"):
            a(n, all=n)
        a("snowy_grass", top="snow", bottom="dirt", side="snow_side")
        a("log", top="log_top", side="log_side")
        a("leaves", all="leaves", flags=CUTOUT)
        a("spruce_leaves", all="spruce_leaves", flags=CUTOUT)
        a("water", all="water", flags=WATER, ticks=6)
        a("lava", all="lava", flags=SOLID | BRIGHT, emit=15, ticks=8)
        for n in textures.CONCRETE:
            a("c_" + n, all="c_" + n)
        a("glowstone", all="glowstone", flags=SOLID | BRIGHT, emit=15)
        a("lantern", all="lantern", flags=SOLID | BRIGHT, emit=15)
        for n in ("neon_magenta", "neon_cyan", "neon_lime", "neon_gold"):
            a(n, all=n, flags=SOLID | BRIGHT, emit=11)
        a("window_lit", all="window_lit", flags=SOLID | BRIGHT, emit=9)
        a("tnt", top="tnt_top", side="tnt_side")
        a("washer", top="washer_top", side="washer_side", front="washer_front")
        a("server_lo", top="server_top", side="server_side", front="server_front")
        a("server_hi", top="server_top", side="server_side", front="server_front_top")
        a("shelf", all="planks", front="shelf_jars")
        a("barrel", top="log_top", side="barrel_side")
        a("pipe_h", all="pipe_h", flags=CUTOUT)
        a("pipe_v", all="pipe_v", flags=CUTOUT)
        a("cobweb", all="cobweb", flags=CUTOUT)
        a("bulb", all="bulb", flags=SOLID | BRIGHT, emit=14)
        a("wire", all="wire", flags=CUTOUT)
        a("asphalt_line", all="asphalt", top="asphalt_line")
        a("cactus", top="c_green", side="cactus_side")
        a("red_light", all="red_light", flags=SOLID | BRIGHT, emit=9)
        a("antenna", all="antenna", flags=CUTOUT)

    def table_bytes(self):
        out = bytearray()
        for i in range(256):
            b = self.blocks[i] if i < len(self.blocks) else self.blocks[0]
            out += struct.pack("<6H8B", *b["tex"], b["flags"], b["emit"], b["frames"], b["ticks"], b["mask"], 0, 0, 0)
        return bytes(out)


class World:
    def __init__(self, pal, X, Y, Z):
        self.pal, self.X, self.Y, self.Z = pal, X, Y, Z
        self.v = np.zeros((Y, Z, X), np.uint8)
        self.objects = []

    def b(self, name):
        return self.pal.id[name]

    def set(self, x, y, z, name):
        if 0 <= x < self.X and 0 <= y < self.Y and 0 <= z < self.Z:
            self.v[y, z, x] = self.b(name) if isinstance(name, str) else name

    def get(self, x, y, z):
        return self.v[y, z, x] if 0 <= x < self.X and 0 <= y < self.Y and 0 <= z < self.Z else 0

    def box(self, x0, y0, z0, x1, y1, z1, name):
        """Fill an inclusive box (coordinates are clipped to the grid)."""
        x0, x1 = sorted((x0, x1)); y0, y1 = sorted((y0, y1)); z0, z1 = sorted((z0, z1))
        self.v[max(y0, 0):y1 + 1, max(z0, 0):z1 + 1, max(x0, 0):x1 + 1] = self.b(name) if isinstance(name, str) else name

    def add_object(self, names):
        """names[y][z][x] block names (or None); returns the object index."""
        arr = np.array([[[self.b(n) if n else 0 for n in row] for row in layer] for layer in names], np.uint8)
        self.objects.append(arr)
        return len(self.objects) - 1

    def save(self, path):
        with open(path, "wb") as f:
            f.write(b"VOX1")
            f.write(struct.pack("<5i", self.X, self.Y, self.Z, len(self.pal.frames), 16))
            f.write(self.pal.table_bytes())
            for fr in self.pal.frames:
                f.write(np.ascontiguousarray(fr, np.uint8).tobytes())
            f.write(np.ascontiguousarray(self.v).tobytes())
            f.write(struct.pack("<i", len(self.objects)))
            for o in self.objects:
                h, d, w = o.shape
                f.write(struct.pack("<3i", w, h, d))
                f.write(np.ascontiguousarray(o).tobytes())


LOOKS = {
    "day": dict(ZEN=(0.30, 0.52, 0.98), HOR=(0.70, 0.84, 1.0), FOG=(0.72, 0.85, 1.0), FOG0=100, FOG1=190,
                SUN=(0.35, 0.72, 0.60), SUNCOL=(1.0, 0.98, 0.86), SUNSIZE=0.055, SKYL=(1.0, 1.0, 1.0),
                BLKL=(1.0, 0.86, 0.62), AMB=0.05, EXPO=1.0, CLOUDY=118, CLOUDA=0.92, STARS=0),
    "sunset": dict(ZEN=(0.24, 0.30, 0.62), HOR=(1.0, 0.56, 0.26), FOG=(0.98, 0.62, 0.36), FOG0=90, FOG1=180,
                   SUN=(-0.25, 0.16, 0.95), SUNCOL=(1.0, 0.86, 0.52), SUNSIZE=0.075, SKYL=(1.0, 0.80, 0.62),
                   BLKL=(1.0, 0.86, 0.62), AMB=0.05, EXPO=1.0, CLOUDY=120, CLOUDA=0.8, STARS=0),
    "night": dict(ZEN=(0.01, 0.02, 0.07), HOR=(0.05, 0.08, 0.20), FOG=(0.04, 0.06, 0.16), FOG0=70, FOG1=160,
                  SUN=(0.3, 0.55, 0.78), SUNCOL=(0.92, 0.95, 1.0), SUNSIZE=0.045, SKYL=(0.20, 0.24, 0.40),
                  BLKL=(1.0, 0.95, 0.90), AMB=0.04, EXPO=1.0, CLOUDY=0, CLOUDA=0, STARS=0.9),
    "nether": dict(ZEN=(0.20, 0.03, 0.02), HOR=(0.42, 0.08, 0.04), FOG=(0.40, 0.07, 0.04), FOG0=40, FOG1=120,
                   SUN=(0, 1, 0), SUNCOL=(0, 0, 0), SUNSIZE=0, SKYL=(0.66, 0.40, 0.32), BLKL=(1.0, 0.80, 0.52),
                   AMB=0.10, EXPO=1.0, CLOUDY=0, CLOUDA=0, STARS=0),
    "end": dict(ZEN=(0.03, 0.01, 0.06), HOR=(0.10, 0.04, 0.16), FOG=(0.08, 0.03, 0.13), FOG0=70, FOG1=170,
                SUN=(0, 1, 0), SUNCOL=(0, 0, 0), SUNSIZE=0, SKYL=(0.62, 0.55, 0.72), BLKL=(0.95, 0.80, 1.0),
                AMB=0.10, EXPO=1.0, CLOUDY=0, CLOUDA=0, STARS=0.6),
    "snow": dict(ZEN=(0.42, 0.58, 0.86), HOR=(0.86, 0.90, 0.97), FOG=(0.86, 0.90, 0.97), FOG0=70, FOG1=170,
                 SUN=(-0.45, 0.42, 0.78), SUNCOL=(1.0, 0.98, 0.92), SUNSIZE=0.05, SKYL=(0.96, 0.98, 1.0),
                 BLKL=(1.0, 0.86, 0.62), AMB=0.06, EXPO=1.0, CLOUDY=124, CLOUDA=0.85, STARS=0),
    "basement": dict(ZEN=(0.20, 0.36, 0.66), HOR=(0.47, 0.56, 0.67), FOG=(0.02, 0.02, 0.03), FOG0=150, FOG1=200,
                     SUN=(0, 1, 0), SUNCOL=(0, 0, 0), SUNSIZE=0, SKYL=(0.60, 0.70, 0.86), BLKL=(1.0, 0.84, 0.60),
                     AMB=0.075, EXPO=1.25, GAMMA=1.55, CLOUDY=0, CLOUDA=0, STARS=0),
}


def frames(n, look="day", **over):
    """(n, NF) parameter block pre-filled with a look; camera/time are filled by the caller."""
    f = np.zeros((n, NF), np.float32)
    vals = dict(LOOKS[look])
    vals.update(over)
    for k, v in vals.items():
        v = np.atleast_1d(np.asarray(v, np.float32))
        if k == "SUN":
            v = v / (np.linalg.norm(v) + 1e-9)
        f[:, P[k]:P[k] + len(v)] = v
    f[:, P["FOV"]] = 72
    f[:, P["FLOODY"]] = -1e9
    f[:, P["WTINT"]:P["WTINT"] + 3] = (0.16, 0.36, 0.72)
    f[:, P["WDENS"]] = 0.10
    return f


def set_cam(f, pos, yaw, pitch, roll=0.0, fov=None, t=None):
    f[:, 0:3] = pos
    f[:, P["YAW"]] = yaw
    f[:, P["PITCH"]] = pitch
    f[:, P["ROLL"]] = roll
    if fov is not None:
        f[:, P["FOV"]] = fov
    if t is not None:
        f[:, P["TIME"]] = t


def look_at(pos, target):
    """yaw, pitch arrays that aim from pos to target (both (n,3) or (3,))."""
    d = np.asarray(target, np.float64) - np.asarray(pos, np.float64)
    yaw = np.arctan2(d[..., 0], d[..., 2])
    pitch = np.arctan2(d[..., 1], np.hypot(d[..., 0], d[..., 2]))
    return yaw, pitch


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def set_obj(f, k, pos, R=None, phase=0.0):
    """Place movable object k: pos = world position of its bottom centre, R = 3x3 rotation."""
    o = P["OBJ"] + k * OBJ_NF
    f[..., o:o + 3] = pos
    f[..., o + 3:o + 12] = (np.eye(3) if R is None else np.asarray(R)).reshape(-1, 9) if np.ndim(R) == 3 else (np.eye(3) if R is None else np.asarray(R)).reshape(9)
    f[..., o + 12] = phase
    f[..., P["NOBJ"]] = np.maximum(f[..., P["NOBJ"]], k + 1)


def render(scene_path, f, out_path, W=1080, H=1920, spp=2, fps=60, crf=10, threads=2):
    """Render a parameter block to an mp4 (or a single png when out_path ends with .png)."""
    build_binary()
    fr_path = os.path.splitext(out_path)[0] + ".frames.bin"
    np.ascontiguousarray(f, np.float32).tofile(fr_path)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    vox = subprocess.Popen([VOX_BIN, scene_path, fr_path, str(W), str(H), str(spp)], stdout=subprocess.PIPE, env=env)
    if out_path.endswith(".png"):
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-i", "-",
               "-frames:v", "1", out_path]
    else:
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps),
               "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p", out_path]
    enc = subprocess.run(cmd, stdin=vox.stdout)
    vox.stdout.close()
    if vox.wait() != 0 or enc.returncode != 0:
        raise RuntimeError(f"render failed for {out_path}")
    os.remove(fr_path)
    return out_path
