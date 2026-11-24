// vox.c — small CPU voxel raycaster with a Minecraft-like look (original engine, no game assets).
//
//   vox scene.bin frames.bin W H spp [first_frame] [count]  > raw rgb24 frames on stdout
//
// Scene: voxel grid + block table + 16x16 texture atlas (+ optional small movable voxel objects).
// Frames: NF float32 per frame (camera, sky, fog, lights, flood plane, object transforms).
// Lighting is baked like the game does it: flood-filled sky/block light (0..15), per-face shade,
// smooth per-vertex light and ambient occlusion.
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NF 112
enum { P_CAM = 0, P_YAW = 3, P_PITCH = 4, P_ROLL = 5, P_FOV = 6, P_TIME = 7, P_ZEN = 8, P_HOR = 11, P_FOG = 14,
       P_FOG0 = 17, P_FOG1 = 18, P_SUN = 19, P_SUNCOL = 22, P_SUNSIZE = 25, P_SKYL = 26, P_BLKL = 29, P_AMB = 32,
       P_EXPO = 33, P_GAMMA = 34, P_CLOUDY = 35, P_CLOUDA = 36, P_STARS = 37, P_FLOODY = 38, P_FLOODR = 39, P_WTINT = 43,
       P_WDENS = 46, P_LAMP = 47, P_SPEC = 50, P_NOBJ = 52, P_OBJ = 56 };
#define OBJ_NF 13 /* pos3, rot9 (row-major, columns = local axes), anim phase */
#define MAX_OBJ 4

#define F_SOLID 1
#define F_CUTOUT 2
#define F_WATER 4
#define F_BRIGHT 8

typedef struct { uint16_t tex[6]; uint8_t flags, emit, anim_frames, anim_ticks, anim_mask, pad[3]; } Block;
typedef struct { int w, h, d; uint8_t v[512]; } Obj;

static int X, Y, Z, NT, TS, BX, BY, BZ, NOBJ;
static uint8_t *vox, *skyl, *blkl, *occ, *tex;
static Block blocks[256];
static Obj objs[MAX_OBJ];
static float lcurve[17];
static const int MIP_OFF[5] = {0, 256, 320, 336, 340};
#define MIP_TEXELS 341

static inline int inb(int x, int y, int z) { return (unsigned)x < (unsigned)X && (unsigned)y < (unsigned)Y && (unsigned)z < (unsigned)Z; }
static inline size_t vidx(int x, int y, int z) { return (size_t)x + (size_t)X * ((size_t)z + (size_t)Z * (size_t)y); }
static inline uint8_t vget(int x, int y, int z) { return inb(x, y, z) ? vox[vidx(x, y, z)] : 0; }
static inline int opaque(int x, int y, int z) { return blocks[vget(x, y, z)].flags & F_SOLID; }
static inline float clampf(float v, float a, float b) { return v < a ? a : (v > b ? b : v); }
static inline float lerpf(float a, float b, float t) { return a + (b - a) * t; }
static inline float fracf(float v) { return v - floorf(v); }
static inline uint32_t hash3(int x, int y, int z) {
    uint32_t h = (uint32_t)x * 374761393u + (uint32_t)y * 668265263u + (uint32_t)z * 2147483647u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return h ^ (h >> 16);
}
static inline float hashf(int x, int y, int z) { return (hash3(x, y, z) & 0xFFFFFF) / 16777216.0f; }

static float vnoise(float x, float y, float z) {
    int xi = (int)floorf(x), yi = (int)floorf(y), zi = (int)floorf(z);
    float fx = x - xi, fy = y - yi, fz = z - zi;
    fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy); fz = fz * fz * (3 - 2 * fz);
    float c[2][2];
    for (int j = 0; j < 2; j++)
        for (int k = 0; k < 2; k++)
            c[j][k] = lerpf(hashf(xi, yi + j, zi + k), hashf(xi + 1, yi + j, zi + k), fx);
    return lerpf(lerpf(c[0][0], c[1][0], fy), lerpf(c[0][1], c[1][1], fy), fz);
}

static void tex_fetch(int t, int lvl, float u, float v, float out[4]) {
    int s = TS >> lvl;
    int iu = (int)(u * s), iv = (int)(v * s);
    iu = iu < 0 ? 0 : (iu >= s ? s - 1 : iu);
    iv = iv < 0 ? 0 : (iv >= s ? s - 1 : iv);
    const uint8_t *p = tex + ((size_t)t * MIP_TEXELS + MIP_OFF[lvl] + iv * s + iu) * 4;
    out[0] = p[0] * (1 / 255.f); out[1] = p[1] * (1 / 255.f); out[2] = p[2] * (1 / 255.f); out[3] = p[3] * (1 / 255.f);
}
static void tex_sample(int t, float lod, float u, float v, float out[4]) {
    lod = clampf(lod, 0, 4);
    int l0 = (int)lod;
    float f = lod - l0;
    tex_fetch(t, l0, u, v, out);
    if (f > 0.02f && l0 < 4) {
        float b[4];
        tex_fetch(t, l0 + 1, u, v, b);
        for (int i = 0; i < 4; i++) out[i] = lerpf(out[i], b[i], f);
    }
}
static inline void face_uv(int face, const float p[3], float *u, float *v) {
    float fx = fracf(p[0]), fy = fracf(p[1]), fz = fracf(p[2]);
    switch (face) {
    case 0: *u = fz; *v = 1 - fy; break;
    case 1: *u = 1 - fz; *v = 1 - fy; break;
    case 2: *u = fx; *v = fz; break;
    case 3: *u = fx; *v = 1 - fz; break;
    case 4: *u = 1 - fx; *v = 1 - fy; break;
    default: *u = fx; *v = 1 - fy; break;
    }
}
static inline int block_tex(const Block *B, int face, float phase) {
    int t = B->tex[face];
    if (B->anim_frames > 1 && (B->anim_mask >> face & 1)) t += ((int)floorf(phase)) % B->anim_frames;
    return t;
}

typedef struct {
    int hit; float t; int face, x, y, z; uint8_t id;
    int whit; float wt; int wx, wy, wz;
    float wlen;
} Hit;

static void trace(const float o[3], const float d[3], float tmax, Hit *h) {
    memset(h, 0, sizeof *h);
    int c[3], step[3];
    float inv[3], tD[3], tM[3];
    for (int a = 0; a < 3; a++) {
        c[a] = (int)floorf(o[a]);
        step[a] = d[a] > 0 ? 1 : -1;
        inv[a] = fabsf(d[a]) > 1e-9f ? 1.0f / d[a] : (d[a] >= 0 ? 1e30f : -1e30f);
        tD[a] = fabsf(inv[a]);
        tM[a] = (d[a] > 0 ? (c[a] + 1 - o[a]) : (o[a] - c[a])) * tD[a];
    }
    float t = 0, wstart = 0;
    int face = -1, inw = 0, first = 1;
    if (inb(c[0], c[1], c[2]) && (blocks[vox[vidx(c[0], c[1], c[2])]].flags & F_WATER)) inw = 1;
    for (int iter = 0; iter < 4096; iter++) {
        if (!inb(c[0], c[1], c[2])) break;
        if (!occ[(c[0] >> 3) + BX * ((c[2] >> 3) + BZ * (c[1] >> 3))]) {
            if (inw) { h->wlen += t - wstart; inw = 0; }
            float tb = 1e30f;
            int ax = 0;
            for (int a = 0; a < 3; a++) {
                int bc = c[a] >> 3;
                float ta = ((d[a] > 0 ? (bc + 1) * 8 : bc * 8) - o[a]) * inv[a];
                if (ta < tb) { tb = ta; ax = a; }
            }
            if (tb > tmax) { t = tmax; break; }
            int bc = c[ax] >> 3;
            t = tb;
            for (int a = 0; a < 3; a++) c[a] = (int)floorf(o[a] + d[a] * t);
            c[ax] = d[ax] > 0 ? (bc + 1) * 8 : bc * 8 - 1;
            for (int a = 0; a < 3; a++) tM[a] = (d[a] > 0 ? (c[a] + 1 - o[a]) : (o[a] - c[a])) * tD[a];
            face = ax * 2 + (d[ax] > 0 ? 1 : 0);
            first = 0;
            continue;
        }
        uint8_t id = vox[vidx(c[0], c[1], c[2])];
        if (id && !first) {
            const Block *B = &blocks[id];
            if (B->flags & F_WATER) {
                if (!inw) {
                    inw = 1; wstart = t;
                    if (!h->whit) { h->whit = 1; h->wt = t; h->wx = c[0]; h->wy = c[1]; h->wz = c[2]; }
                }
            } else {
                int pass = 0;
                if (B->flags & F_CUTOUT) {
                    float p[3] = {o[0] + d[0] * t, o[1] + d[1] * t, o[2] + d[2] * t}, u, v, px[4];
                    p[face >> 1] = (float)c[face >> 1] + 0.5f;
                    face_uv(face, p, &u, &v);
                    tex_fetch(block_tex(B, face, 0), 0, u, v, px);
                    pass = px[3] < 0.5f;
                }
                if (!pass) {
                    if (inw) h->wlen += t - wstart;
                    h->hit = 1; h->t = t; h->face = face; h->x = c[0]; h->y = c[1]; h->z = c[2]; h->id = id;
                    return;
                }
                if (inw) { h->wlen += t - wstart; inw = 0; }
            }
        } else if (!id && inw) { h->wlen += t - wstart; inw = 0; }
        first = 0;
        int a = (tM[0] < tM[1]) ? (tM[0] < tM[2] ? 0 : 2) : (tM[1] < tM[2] ? 1 : 2);
        t = tM[a];
        if (t > tmax) { t = tmax; break; }
        tM[a] += tD[a];
        c[a] += step[a];
        face = a * 2 + (step[a] > 0 ? 1 : 0);
    }
    if (inw) h->wlen += t - wstart;
    h->t = tmax;
}

static inline void cell_light(int x, int y, int z, float *s, float *b) {
    if (!inb(x, y, z)) { *s = y < 0 ? 0 : 15; *b = 0; return; }
    size_t i = vidx(x, y, z);
    *s = skyl[i]; *b = blkl[i];
}
static inline float lc(float l) {
    l = clampf(l, 0, 15);
    int i = (int)l;
    return lerpf(lcurve[i], lcurve[i + 1], l - i);
}

// Smooth light + AO for a face of the static grid. p = hit point.
static void face_light(int x, int y, int z, int face, const float p[3], float *sky, float *blk, float *ao) {
    static const float AO[4] = {1.0f, 0.80f, 0.62f, 0.46f};
    int ax = face >> 1, n[3] = {0, 0, 0};
    n[ax] = (face & 1) ? -1 : 1;
    int ua = (ax + 1) % 3, va = (ax + 2) % 3;
    int c[3] = {x + n[0], y + n[1], z + n[2]};
    float fu = fracf(p[ua]), fv = fracf(p[va]);
    float s0, b0;
    cell_light(c[0], c[1], c[2], &s0, &b0);
    float S = 0, B = 0, A = 0;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++) {
            int du[3] = {0, 0, 0}, dv[3] = {0, 0, 0};
            du[ua] = i ? 1 : -1;
            dv[va] = j ? 1 : -1;
            int o1 = opaque(c[0] + du[0], c[1] + du[1], c[2] + du[2]) != 0;
            int o2 = opaque(c[0] + dv[0], c[1] + dv[1], c[2] + dv[2]) != 0;
            int o3 = opaque(c[0] + du[0] + dv[0], c[1] + du[1] + dv[1], c[2] + du[2] + dv[2]) != 0;
            float s1 = s0, b1 = b0, s2 = s0, b2 = b0, s3 = s0, b3 = b0;
            if (!o1) cell_light(c[0] + du[0], c[1] + du[1], c[2] + du[2], &s1, &b1);
            if (!o2) cell_light(c[0] + dv[0], c[1] + dv[1], c[2] + dv[2], &s2, &b2);
            if (!o3 && !(o1 && o2)) cell_light(c[0] + du[0] + dv[0], c[1] + du[1] + dv[1], c[2] + du[2] + dv[2], &s3, &b3);
            float w = (i ? fu : 1 - fu) * (j ? fv : 1 - fv);
            S += w * (s0 + s1 + s2 + s3) * 0.25f;
            B += w * (b0 + b1 + b2 + b3) * 0.25f;
            A += w * AO[(o1 && o2) ? 3 : o1 + o2 + o3];
        }
    *sky = S; *blk = B; *ao = A;
}

static void sky_color(const float *P, const float d[3], float out[3]) {
    float h = d[1];
    float k = h > 0 ? powf(h, 0.5f) : 0;
    for (int i = 0; i < 3; i++) out[i] = lerpf(P[P_HOR + i], P[P_ZEN + i], k);
    if (h < 0) {
        float m = clampf(-h * 5.0f, 0, 1);
        for (int i = 0; i < 3; i++) out[i] = lerpf(out[i], P[P_FOG + i], m);
    }
    if (P[P_STARS] > 0 && h > 0.02f) {
        float ax = fabsf(d[0]), ay = fabsf(d[1]), az = fabsf(d[2]), a, b;
        int m = ax > ay ? (ax > az ? 0 : 2) : (ay > az ? 1 : 2);
        if (m == 0) { a = d[1] / ax; b = d[2] / ax; } else if (m == 1) { a = d[0] / ay; b = d[2] / ay; } else { a = d[0] / az; b = d[1] / az; }
        int ia = (int)floorf(a * 90), ib = (int)floorf(b * 90);
        float r = hashf(ia, ib, m * 7 + (d[m] > 0));
        if (r < 0.012f) {
            float br = P[P_STARS] * (0.45f + 0.55f * hashf(ib, ia, 3)) * clampf(h * 6, 0, 1);
            for (int i = 0; i < 3; i++) out[i] += br;
        }
    }
    if (P[P_SUNSIZE] > 0) {
        const float *s = P + P_SUN;
        float sd = d[0] * s[0] + d[1] * s[1] + d[2] * s[2];
        if (sd > 0) {
            float su[3] = {s[2], 0, -s[0]};
            float l = sqrtf(su[0] * su[0] + su[2] * su[2]);
            if (l < 1e-4f) { su[0] = 1; su[2] = 0; l = 1; }
            su[0] /= l; su[2] /= l;
            float sv[3] = {su[1] * s[2] - su[2] * s[1], su[2] * s[0] - su[0] * s[2], su[0] * s[1] - su[1] * s[0]};
            float a = (d[0] * su[0] + d[2] * su[2]) / sd, b = (d[0] * sv[0] + d[1] * sv[1] + d[2] * sv[2]) / sd;
            float sz = P[P_SUNSIZE];
            if (fabsf(a) < sz && fabsf(b) < sz) {
                for (int i = 0; i < 3; i++) out[i] = P[P_SUNCOL + i];
            } else {
                float g = expf(-(a * a + b * b) / (sz * sz * 10.0f));
                for (int i = 0; i < 3; i++) out[i] += P[P_SUNCOL + i] * g * 0.45f;
            }
        }
    }
}

static int trace_obj(const Obj *ob, const float *T, const float o[3], const float d[3], float tmax,
                     float *tout, int *face_out, int cell[3], float lp[3]) {
    float rel[3] = {o[0] - T[0], o[1] - T[1], o[2] - T[2]}, lo[3], ld[3];
    const float *R = T + 3;
    float piv[3] = {ob->w * 0.5f, 0, ob->d * 0.5f};
    int dim[3] = {ob->w, ob->h, ob->d};
    for (int i = 0; i < 3; i++) {
        lo[i] = R[i] * rel[0] + R[3 + i] * rel[1] + R[6 + i] * rel[2] + piv[i];
        ld[i] = R[i] * d[0] + R[3 + i] * d[1] + R[6 + i] * d[2];
    }
    float t0 = 0, t1 = tmax;
    int eax = -1;
    for (int a = 0; a < 3; a++) {
        if (fabsf(ld[a]) < 1e-9f) {
            if (lo[a] < 0 || lo[a] > dim[a]) return 0;
        } else {
            float ta = (0 - lo[a]) / ld[a], tb = (dim[a] - lo[a]) / ld[a];
            if (ta > tb) { float s = ta; ta = tb; tb = s; }
            if (ta > t0) { t0 = ta; eax = a; }
            if (tb < t1) t1 = tb;
            if (t0 > t1) return 0;
        }
    }
    int c[3], step[3];
    float tD[3], tM[3], t = t0;
    for (int a = 0; a < 3; a++) {
        float p = lo[a] + ld[a] * (t0 + 1e-4f);
        c[a] = (int)floorf(p);
        if (c[a] < 0) c[a] = 0;
        if (c[a] >= dim[a]) c[a] = dim[a] - 1;
        step[a] = ld[a] > 0 ? 1 : -1;
        tD[a] = fabsf(ld[a]) > 1e-9f ? fabsf(1.0f / ld[a]) : 1e30f;
        tM[a] = (ld[a] > 0 ? (c[a] + 1 - lo[a]) : (lo[a] - c[a])) * tD[a];
    }
    int face = eax < 0 ? -1 : eax * 2 + (ld[eax] > 0 ? 1 : 0);
    for (int iter = 0; iter < 64; iter++) {
        uint8_t id = ob->v[c[0] + ob->w * (c[2] + ob->d * c[1])];
        if (id && face >= 0) {
            *tout = t; *face_out = face;
            cell[0] = c[0]; cell[1] = c[1]; cell[2] = c[2];
            for (int i = 0; i < 3; i++) lp[i] = lo[i] + ld[i] * t;
            return id;
        }
        int a = (tM[0] < tM[1]) ? (tM[0] < tM[2] ? 0 : 2) : (tM[1] < tM[2] ? 1 : 2);
        t = tM[a];
        if (t > t1 + 1e-4f) return 0;
        tM[a] += tD[a];
        c[a] += step[a];
        if (c[a] < 0 || c[a] >= dim[a]) return 0;
        face = a * 2 + (step[a] > 0 ? 1 : 0);
    }
    return 0;
}

static void apply_light(const float *P, float alb[3], float shade, float ao, float sky, float blk, float out[3]) {
    float bs = lc(sky), bb = lc(blk);
    for (int i = 0; i < 3; i++) {
        float l = P[P_SKYL + i] * bs, l2 = P[P_BLKL + i] * bb;
        if (l2 > l) l = l2;
        if (l < P[P_AMB]) l = P[P_AMB];
        out[i] = alb[i] * shade * ao * l;
    }
}

static void shade_ray(const float *P, const float o[3], const float d[3], float pixang, float out[3]) {
    static const float FSH[6] = {0.62f, 0.62f, 1.0f, 0.5f, 0.8f, 0.8f};
    float tmax = P[P_FOG1] > 0 ? P[P_FOG1] : 400.0f;
    float time = P[P_TIME];
    Hit h;
    trace(o, d, tmax, &h);
    float col[3], t = h.t;
    int have = 0;
    // movable objects
    int nobj = (int)P[P_NOBJ];
    for (int k = 0; k < nobj && k < NOBJ; k++) {
        const float *T = P + P_OBJ + k * OBJ_NF;
        float to, lp[3];
        int fo, cc[3];
        int id = trace_obj(&objs[k], T, o, d, h.hit ? h.t : tmax, &to, &fo, cc, lp);
        if (id && to < t) {
            const Block *B = &blocks[id];
            float u, v, px[4];
            float q[3] = {lp[0], lp[1], lp[2]};
            q[fo >> 1] = cc[fo >> 1] + 0.5f;
            face_uv(fo, q, &u, &v);
            float lod = log2f(fmaxf(to * pixang * TS, 1e-3f)) - 0.3f;
            tex_sample(block_tex(B, fo, T[12]), lod, u, v, px);
            const float *R = T + 3;
            int ax = fo >> 1;
            float sg = (fo & 1) ? -1.f : 1.f, n[3] = {R[ax] * sg, R[3 + ax] * sg, R[6 + ax] * sg};
            float shade = n[0] * n[0] * 0.62f + n[2] * n[2] * 0.8f + n[1] * n[1] * (n[1] > 0 ? 1.0f : 0.5f);
            float wp[3] = {o[0] + d[0] * to + n[0] * 0.5f, o[1] + d[1] * to + n[1] * 0.5f, o[2] + d[2] * to + n[2] * 0.5f}, s, b;
            cell_light((int)floorf(wp[0]), (int)floorf(wp[1]), (int)floorf(wp[2]), &s, &b);
            if (B->flags & F_BRIGHT) { col[0] = px[0]; col[1] = px[1]; col[2] = px[2]; }
            else apply_light(P, px, shade, 1.0f, s, b, col);
            // emissive texels (alpha < 1 marks "glow" pixels on object textures)
            if (px[3] < 0.99f) { col[0] = px[0]; col[1] = px[1]; col[2] = px[2]; }
            t = to; have = 1; h.hit = 1;
        }
    }
    if (!have && h.hit) {
        const Block *B = &blocks[h.id];
        float p[3] = {o[0] + d[0] * t, o[1] + d[1] * t, o[2] + d[2] * t}, u, v, px[4];
        p[h.face >> 1] = (float)(h.face & 1 ? ((h.face >> 1) == 0 ? h.x : (h.face >> 1) == 1 ? h.y : h.z)
                                             : ((h.face >> 1) == 0 ? h.x : (h.face >> 1) == 1 ? h.y : h.z) + 1);
        float q[3] = {p[0], p[1], p[2]};
        q[h.face >> 1] = ((h.face >> 1) == 0 ? h.x : (h.face >> 1) == 1 ? h.y : h.z) + 0.5f;
        face_uv(h.face, q, &u, &v);
        float cosi = fabsf(d[h.face >> 1]);
        float lod = log2f(fmaxf(t * pixang * TS / fmaxf(cosi, 0.25f), 1e-3f)) - 0.3f;
        tex_sample(block_tex(B, h.face, time * 20.0f / (B->anim_ticks ? B->anim_ticks : 1)), lod, u, v, px);
        if (B->flags & F_BRIGHT) { col[0] = px[0]; col[1] = px[1]; col[2] = px[2]; }
        else {
            float s, b, ao;
            face_light(h.x, h.y, h.z, h.face, q, &s, &b, &ao);
            apply_light(P, px, FSH[h.face], ao, s, b, col);
        }
        have = 1;
    }
    if (!have) sky_color(P, d, col);
    // clouds (flat layer)
    if (P[P_CLOUDY] > 0 && fabsf(d[1]) > 1e-4f) {
        float tc = (P[P_CLOUDY] - o[1]) / d[1];
        if (tc > 0 && (!have || tc < t)) {
            float cx = (o[0] + d[0] * tc + time * 1.5f) / 12.0f, cz = (o[2] + d[2] * tc) / 12.0f;
            int ix = (int)floorf(cx), iz = (int)floorf(cz);
            float dens = vnoise(ix * 0.37f, 3.1f, iz * 0.37f) * 0.65f + hashf(ix, 91, iz) * 0.35f;
            if (dens > 0.56f) {
                float a = P[P_CLOUDA] * clampf(1.0f - tc / 900.0f, 0, 1);
                float w = d[1] > 0 ? 0.90f : 1.0f;
                for (int i = 0; i < 3; i++) col[i] = lerpf(col[i], w * clampf(P[P_SKYL + i] * 1.05f + 0.08f, 0, 1), a);
            }
        }
    }
    // static water blocks
    if (h.whit && (!have || h.wt <= t + 1e-3f)) {
        float fog = 1.0f - expf(-h.wlen * P[P_WDENS]);
        for (int i = 0; i < 3; i++) col[i] = lerpf(col[i], P[P_WTINT + i] * 0.55f, fog);
        float p[3] = {o[0] + d[0] * h.wt, o[1] + d[1] * h.wt, o[2] + d[2] * h.wt};
        float nz = vnoise(p[0] * 0.9f + time * 0.7f, time * 0.5f, p[2] * 0.9f) - 0.5f;
        float s, b;
        cell_light(h.wx, h.wy + 1, h.wz, &s, &b);
        float l = fmaxf(lc(s) * (P[P_SKYL] + P[P_SKYL + 1] + P[P_SKYL + 2]) / 3, lc(b));
        float fres = 0.35f + 0.45f * powf(1.0f - fabsf(d[1]), 3.0f);
        float sk[3], rd[3] = {d[0], fabsf(d[1]), d[2]};
        sky_color(P, rd, sk);
        for (int i = 0; i < 3; i++) {
            float wc = (P[P_WTINT + i] * (0.9f + nz * 0.5f)) * l * 0.6f + sk[i] * 0.4f * l;
            col[i] = lerpf(col[i], wc, fres);
        }
        if (!have || h.wt < t) t = have ? t : h.wt;
        have = 1;
    }
    // rising flood plane inside a region (basement)
    float fy = P[P_FLOODY];
    if (fy > -1e8f) {
        const float *r = P + P_FLOODR;
        int cam_in = o[1] < fy && o[0] > r[0] && o[0] < r[1] && o[2] > r[2] && o[2] < r[3];
        float tend = have ? t : tmax, tw = -1;
        if (fabsf(d[1]) > 1e-5f) {
            tw = (fy - o[1]) / d[1];
            float px = o[0] + d[0] * tw, pz = o[2] + d[2] * tw;
            if (tw <= 0 || tw >= tend || px < r[0] || px > r[1] || pz < r[2] || pz > r[3]) tw = -1;
        }
        if (cam_in) {
            float under = tw > 0 ? tw : tend;
            if (tw > 0) {
                float nz = vnoise((o[0] + d[0] * tw) * 1.3f + time * 0.6f, time * 0.8f, (o[2] + d[2] * tw) * 1.3f);
                for (int i = 0; i < 3; i++) col[i] = lerpf(col[i], P[P_WTINT + i] * (0.8f + nz * 0.9f), 0.45f);
            }
            float cau = 0.85f + 0.3f * vnoise((o[0] + d[0] * under) * 1.7f + time * 0.9f, (o[1] + d[1] * under) * 1.7f, (o[2] + d[2] * under) * 1.7f + time * 0.7f);
            float fog = 1.0f - expf(-under * P[P_WDENS]);
            for (int i = 0; i < 3; i++) col[i] = lerpf(col[i] * cau, P[P_WTINT + i] * 0.5f, fog);
        } else if (tw > 0) {
            float depth = tend - tw;
            float fog = 1.0f - expf(-depth * P[P_WDENS] * 1.6f);
            for (int i = 0; i < 3; i++) col[i] = lerpf(col[i], P[P_WTINT + i] * 0.4f, fog);
            float wx = o[0] + d[0] * tw, wz = o[2] + d[2] * tw;
            float e = 0.08f;
            float n0 = vnoise(wx * 1.6f + time * 0.5f, time * 0.9f, wz * 1.6f);
            float nx = vnoise((wx + e) * 1.6f + time * 0.5f, time * 0.9f, wz * 1.6f) - n0;
            float nzz = vnoise(wx * 1.6f + time * 0.5f, time * 0.9f, (wz + e) * 1.6f) - n0;
            float n[3] = {-nx * 2.2f, 1, -nzz * 2.2f};
            float nl = sqrtf(n[0] * n[0] + 1 + n[2] * n[2]);
            n[0] /= nl; n[1] /= nl; n[2] /= nl;
            float dn = d[0] * n[0] + d[1] * n[1] + d[2] * n[2];
            float rf[3] = {d[0] - 2 * dn * n[0], d[1] - 2 * dn * n[1], d[2] - 2 * dn * n[2]};
            float L[3] = {P[P_LAMP] - wx, P[P_LAMP + 1] - fy, P[P_LAMP + 2] - wz};
            float ll = sqrtf(L[0] * L[0] + L[1] * L[1] + L[2] * L[2]) + 1e-6f;
            float sp = fmaxf(0, (rf[0] * L[0] + rf[1] * L[1] + rf[2] * L[2]) / ll);
            sp = powf(sp, 48.0f) * P[P_SPEC] / (1.0f + ll * ll * 0.03f);
            float s, b;
            cell_light((int)floorf(wx), (int)floorf(fy + 0.5f), (int)floorf(wz), &s, &b);
            float l = fmaxf(lc(b), lc(s) * 0.8f);
            float fres = 0.30f + 0.5f * powf(1.0f - fabsf(d[1]), 3.0f);
            for (int i = 0; i < 3; i++) {
                float wc = P[P_WTINT + i] * (0.75f + n0 * 0.6f) * l * P[P_BLKL + i];
                col[i] = lerpf(col[i], wc, fres) + sp * P[P_BLKL + i];
            }
        }
    }
    // distance fog
    if (have && P[P_FOG1] > 0) {
        float f = clampf((t - P[P_FOG0]) / (P[P_FOG1] - P[P_FOG0]), 0, 1);
        f = f * f * (3 - 2 * f);
        if (f > 0) {
            float sk[3];
            sky_color(P, d, sk);
            for (int i = 0; i < 3; i++) col[i] = lerpf(col[i], sk[i], f);
        }
    }
    float ig = P[P_GAMMA] > 0 ? 1.0f / P[P_GAMMA] : 1.0f;
    for (int i = 0; i < 3; i++) {
        float c = clampf(col[i] * P[P_EXPO], 0, 1);
        out[i] = ig == 1.0f ? c : powf(c, ig);   // shadow lift for dark interiors
    }
}

static void bake_light(void) {
    size_t n = (size_t)X * Y * Z;
    skyl = calloc(n, 1); blkl = calloc(n, 1);
    int *q = malloc(sizeof(int) * n);
    size_t qh = 0, qt = 0;
    for (int z = 0; z < Z; z++)
        for (int x = 0; x < X; x++) {
            int l = 15;
            for (int y = Y - 1; y >= 0 && l > 0; y--) {
                const Block *B = &blocks[vox[vidx(x, y, z)]];
                if (B->flags & F_SOLID) break;
                if (B->flags & F_WATER) l -= 2;
                if (l <= 0) break;
                skyl[vidx(x, y, z)] = (uint8_t)l;
                q[qt++] = (int)vidx(x, y, z);
            }
        }
    static const int D[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    for (int pass = 0; pass < 2; pass++) {
        uint8_t *L = pass ? blkl : skyl;
        if (pass) {
            qh = qt = 0;
            for (size_t i = 0; i < n; i++)
                if (blocks[vox[i]].emit) { blkl[i] = blocks[vox[i]].emit; q[qt++] = (int)i; }
        }
        while (qh < qt) {
            size_t i = (size_t)q[qh++];
            int l = L[i];
            if (l <= 1) continue;
            int x = (int)(i % X), z = (int)((i / X) % Z), y = (int)(i / ((size_t)X * Z));
            for (int k = 0; k < 6; k++) {
                int nx = x + D[k][0], ny = y + D[k][1], nz = z + D[k][2];
                if (!inb(nx, ny, nz)) continue;
                size_t j = vidx(nx, ny, nz);
                const Block *B = &blocks[vox[j]];
                if (B->flags & F_SOLID) continue;
                int nl = l - 1 - ((B->flags & F_WATER) ? 1 : 0);
                if (nl > L[j]) { L[j] = (uint8_t)nl; q[qt++] = (int)j; }
            }
        }
    }
    free(q);
}

static void die(const char *m) { fprintf(stderr, "vox: %s\n", m); exit(1); }
#define RD(ptr, sz) do { if (fread(ptr, 1, sz, f) != (size_t)(sz)) die("short scene file"); } while (0)

int main(int argc, char **argv) {
    if (argc < 6) die("usage: vox scene.bin frames.bin W H spp [first] [count]");
    FILE *f = fopen(argv[1], "rb");
    if (!f) die("cannot open scene");
    char magic[4];
    RD(magic, 4);
    if (memcmp(magic, "VOX1", 4)) die("bad magic");
    int hdr[5];
    RD(hdr, sizeof hdr);
    X = hdr[0]; Y = hdr[1]; Z = hdr[2]; NT = hdr[3]; TS = hdr[4];
    if (TS != 16) die("texture size must be 16");
    RD(blocks, sizeof blocks);
    tex = malloc((size_t)NT * MIP_TEXELS * 4);
    for (int t = 0; t < NT; t++) {
        uint8_t *m = tex + (size_t)t * MIP_TEXELS * 4;
        RD(m, 256 * 4);
        for (int l = 1; l < 5; l++) {
            int s = TS >> l;
            const uint8_t *src = m + MIP_OFF[l - 1] * 4;
            uint8_t *dst = m + MIP_OFF[l] * 4;
            for (int y = 0; y < s; y++)
                for (int x = 0; x < s; x++)
                    for (int c = 0; c < 4; c++) {
                        int a = src[((2 * y) * (s * 2) + 2 * x) * 4 + c] + src[((2 * y) * (s * 2) + 2 * x + 1) * 4 + c] +
                                src[((2 * y + 1) * (s * 2) + 2 * x) * 4 + c] + src[((2 * y + 1) * (s * 2) + 2 * x + 1) * 4 + c];
                        dst[(y * s + x) * 4 + c] = (uint8_t)((a + 2) / 4);
                    }
        }
    }
    size_t n = (size_t)X * Y * Z;
    vox = malloc(n);
    RD(vox, n);
    RD(&NOBJ, 4);
    if (NOBJ > MAX_OBJ) die("too many objects");
    for (int k = 0; k < NOBJ; k++) {
        int d3[3];
        RD(d3, 12);
        objs[k].w = d3[0]; objs[k].h = d3[1]; objs[k].d = d3[2];
        if (d3[0] * d3[1] * d3[2] > 512) die("object too big");
        RD(objs[k].v, (size_t)d3[0] * d3[1] * d3[2]);
    }
    fclose(f);
    BX = (X + 7) / 8; BY = (Y + 7) / 8; BZ = (Z + 7) / 8;
    occ = calloc((size_t)BX * BY * BZ, 1);
    for (int y = 0; y < Y; y++)
        for (int z = 0; z < Z; z++)
            for (int x = 0; x < X; x++)
                if (vox[vidx(x, y, z)]) occ[(x >> 3) + BX * ((z >> 3) + BZ * (y >> 3))] = 1;
    for (int i = 0; i <= 16; i++) lcurve[i] = i >= 15 ? 1.0f : powf(0.8f, 15 - i);
    bake_light();

    f = fopen(argv[2], "rb");
    if (!f) die("cannot open frames");
    fseek(f, 0, SEEK_END);
    long fsz = ftell(f);
    fseek(f, 0, SEEK_SET);
    int nfr = (int)(fsz / (NF * 4));
    float *FR = malloc((size_t)nfr * NF * 4);
    if (fread(FR, NF * 4, nfr, f) != (size_t)nfr) die("short frames file");
    fclose(f);
    int W = atoi(argv[3]), H = atoi(argv[4]), spp = atoi(argv[5]);
    int first = argc > 6 ? atoi(argv[6]) : 0, count = argc > 7 ? atoi(argv[7]) : nfr - first;
    static const float J1[1][2] = {{0.5f, 0.5f}};
    static const float J2[2][2] = {{0.25f, 0.25f}, {0.75f, 0.75f}};
    static const float J4[4][2] = {{0.375f, 0.125f}, {0.875f, 0.375f}, {0.125f, 0.625f}, {0.625f, 0.875f}};
    const float (*J)[2] = spp >= 4 ? J4 : (spp == 2 ? J2 : J1);
    if (spp >= 4) spp = 4; else if (spp != 2) spp = 1;
    uint8_t *img = malloc((size_t)W * H * 3);
    for (int fi = first; fi < first + count && fi < nfr; fi++) {
        const float *P = FR + (size_t)fi * NF;
        float cy = cosf(P[P_YAW]), sy = sinf(P[P_YAW]), cp = cosf(P[P_PITCH]), sp = sinf(P[P_PITCH]);
        float fw[3] = {sy * cp, sp, cy * cp}, r0[3] = {cy, 0, -sy};
        float u0[3] = {fw[1] * r0[2] - fw[2] * r0[1], fw[2] * r0[0] - fw[0] * r0[2], fw[0] * r0[1] - fw[1] * r0[0]};
        float cr = cosf(P[P_ROLL]), sr = sinf(P[P_ROLL]), rt[3], up[3];
        for (int i = 0; i < 3; i++) { rt[i] = r0[i] * cr + u0[i] * sr; up[i] = u0[i] * cr - r0[i] * sr; }
        float th = tanf(P[P_FOV] * 0.5f * (float)M_PI / 180.0f), asp = (float)W / H;
        float pixang = 2 * th / H;
#pragma omp parallel for schedule(dynamic, 8)
        for (int py = 0; py < H; py++) {
            for (int px = 0; px < W; px++) {
                float acc[3] = {0, 0, 0};
                for (int s = 0; s < spp; s++) {
                    float nx = ((px + J[s][0]) / W * 2 - 1) * th * asp, ny = (1 - (py + J[s][1]) / H * 2) * th;
                    float d[3] = {fw[0] + rt[0] * nx + up[0] * ny, fw[1] + rt[1] * nx + up[1] * ny, fw[2] + rt[2] * nx + up[2] * ny};
                    float l = 1.0f / sqrtf(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
                    d[0] *= l; d[1] *= l; d[2] *= l;
                    float c[3];
                    shade_ray(P, P + P_CAM, d, pixang, c);
                    acc[0] += clampf(c[0], 0, 1); acc[1] += clampf(c[1], 0, 1); acc[2] += clampf(c[2], 0, 1);
                }
                uint8_t *o = img + ((size_t)py * W + px) * 3;
                for (int i = 0; i < 3; i++) o[i] = (uint8_t)(acc[i] / spp * 255.0f + 0.5f);
            }
        }
        if (fwrite(img, 1, (size_t)W * H * 3, stdout) != (size_t)W * H * 3) die("write failed");
    }
    return 0;
}
