#include "fastdec.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define SZFP_X86 1
#include <immintrin.h>
#endif

/* coefficient order of zfp's 3D codec (template/codec3.c) */
#define IDX(i, j, k) ((i) + 4 * ((j) + 4 * (k)))
static const unsigned char PERM[64] = {
    IDX(0, 0, 0), IDX(1, 0, 0), IDX(0, 1, 0), IDX(0, 0, 1), IDX(0, 1, 1), IDX(1, 0, 1),
    IDX(1, 1, 0), IDX(2, 0, 0), IDX(0, 2, 0), IDX(0, 0, 2), IDX(1, 1, 1), IDX(2, 1, 0),
    IDX(2, 0, 1), IDX(0, 2, 1), IDX(1, 2, 0), IDX(1, 0, 2), IDX(0, 1, 2), IDX(3, 0, 0),
    IDX(0, 3, 0), IDX(0, 0, 3), IDX(2, 1, 1), IDX(1, 2, 1), IDX(1, 1, 2), IDX(0, 2, 2),
    IDX(2, 0, 2), IDX(2, 2, 0), IDX(3, 1, 0), IDX(3, 0, 1), IDX(0, 3, 1), IDX(1, 3, 0),
    IDX(1, 0, 3), IDX(0, 1, 3), IDX(1, 2, 2), IDX(2, 1, 2), IDX(2, 2, 1), IDX(3, 1, 1),
    IDX(1, 3, 1), IDX(1, 1, 3), IDX(3, 2, 0), IDX(3, 0, 2), IDX(0, 3, 2), IDX(2, 3, 0),
    IDX(2, 0, 3), IDX(0, 2, 3), IDX(2, 2, 2), IDX(3, 2, 1), IDX(3, 1, 2), IDX(1, 3, 2),
    IDX(2, 3, 1), IDX(2, 1, 3), IDX(1, 2, 3), IDX(0, 3, 3), IDX(3, 0, 3), IDX(3, 3, 0),
    IDX(3, 2, 2), IDX(2, 3, 2), IDX(2, 2, 3), IDX(1, 3, 3), IDX(3, 1, 3), IDX(3, 3, 1),
    IDX(2, 3, 3), IDX(3, 2, 3), IDX(3, 3, 2), IDX(3, 3, 3),
};
#undef IDX

#define NBMASK 0xaaaaaaaau

/* at least 57 valid bits starting at bit pos; buf has 8 bytes of zero padding */
static inline uint64_t peek(const unsigned char* buf, uint32_t pos) {
    uint64_t v;
    memcpy(&v, buf + (pos >> 3), 8);
    return v >> (pos & 7);
}

/* zfp decode_few_ints with ctz-based run scanning; returns bit planes MSB first */
static inline int decode_planes(
    const unsigned char* buf,
    uint32_t pos,
    uint32_t bits,
    uint64_t planes[32]
) {
    uint32_t n = 0;
    int k = 32;
    int np = 0;

    while (bits && k-- > 0) {
        uint32_t m = n < bits ? n : bits;
        uint64_t x;

        bits -= m;
        if (m <= 56) {
            x = m ? peek(buf, pos) & ((1ull << m) - 1) : 0;
            pos += m;
        } else {
            x = peek(buf, pos) & 0xffffffffull;
            x |= (peek(buf, pos + 32) & ((1ull << (m - 32)) - 1)) << 32;
            pos += m;
        }

        while (bits && n < 64) {
            uint32_t lim_all;

            bits--;
            if (!(peek(buf, pos++) & 1)) {
                break;
            }

            /* scan for the next one-bit, reading at most lim_all zeros */
            lim_all = 63 - n;
            if (lim_all > bits) {
                lim_all = bits;
            }
            for (;;) {
                uint64_t w = peek(buf, pos);
                uint32_t lim = lim_all < 56 ? lim_all : 56;
                uint32_t z = w ? (uint32_t)__builtin_ctzll(w) : 64;

                if (z < lim) {
                    n += z;
                    bits -= z + 1;
                    pos += z + 1;
                    break;
                }
                n += lim;
                bits -= lim;
                pos += lim;
                lim_all -= lim;
                if (!lim_all) {
                    break;
                }
            }
            x += 1ull << n;
            n++;
        }
        planes[np++] = x;
    }
    return np;
}

#ifdef SZFP_X86
/* decode_planes with the group tests of a bit plane parsed at once. After the
 * verbatim bits, every newly significant coefficient is coded as a group-test 1,
 * a run of zeros and the coefficient's own 1, and the plane ends with a group-test
 * 0; so in a window that starts with a 1 the set bits alternate between group
 * tests and coefficients until a coefficient bit is followed by a 0. pdep picks
 * every other set bit, the first coefficient bit followed by a 0 ends the plane,
 * and pext of the window without the group-test bits gives the plane's new bits
 * from coefficient n upwards. A plane the window does not hold whole, that would
 * run out of bits, or that could reach coefficient 63, whose one bit is implied,
 * goes through the loop of decode_planes instead. */
__attribute__((target("bmi2,popcnt"))) static int decode_planes_bmi2(
    const unsigned char* buf,
    uint32_t pos,
    uint32_t bits,
    uint64_t planes[32]
) {
    uint32_t n = 0;
    int k = 32;
    int np = 0;

    while (bits && k-- > 0) {
        uint32_t m = n < bits ? n : bits;
        uint64_t x;

        bits -= m;
        if (m <= 56) {
            x = m ? peek(buf, pos) & ((1ull << m) - 1) : 0;
            pos += m;
        } else {
            x = peek(buf, pos) & 0xffffffffull;
            x |= (peek(buf, pos + 32) & ((1ull << (m - 32)) - 1)) << 32;
            pos += m;
        }

        if (bits && n < 64) {
            uint64_t w = peek(buf, pos) & ((1ull << 57) - 1);

            if (!(w & 1)) { /* no new coefficient in this plane */
                bits--;
                pos++;
                planes[np++] = x;
                continue;
            }
            {
                uint64_t g = _pdep_u64(0x5555555555555555ull, w);
                uint64_t s = _pdep_u64(0xAAAAAAAAAAAAAAAAull, w);
                /* a coefficient bit followed by a 0, that 0 inside the window */
                uint64_t end = s & ~(w >> 1) & ((1ull << 56) - 1);

                if (end) {
                    uint32_t e = (uint32_t)__builtin_ctzll(end);
                    uint32_t used = e + 2;
                    uint64_t keep = ((2ull << e) - 1) & ~g;
                    uint32_t add = (uint32_t)__builtin_popcountll(keep);

                    if (used <= bits && n + add < 63) {
                        x |= _pext_u64(w, keep) << n;
                        n += add;
                        bits -= used;
                        pos += used;
                        planes[np++] = x;
                        continue;
                    }
                }
            }
        }

        while (bits && n < 64) {
            uint32_t lim_all;

            bits--;
            if (!(peek(buf, pos++) & 1)) {
                break;
            }

            lim_all = 63 - n;
            if (lim_all > bits) {
                lim_all = bits;
            }
            for (;;) {
                uint64_t w = peek(buf, pos);
                uint32_t lim = lim_all < 56 ? lim_all : 56;
                uint32_t z = w ? (uint32_t)__builtin_ctzll(w) : 64;

                if (z < lim) {
                    n += z;
                    bits -= z + 1;
                    pos += z + 1;
                    break;
                }
                n += lim;
                bits -= lim;
                pos += lim;
                lim_all -= lim;
                if (!lim_all) {
                    break;
                }
            }
            x += 1ull << n;
            n++;
        }
        planes[np++] = x;
    }
    return np;
}
#endif

static void transpose_scalar(const uint64_t* planes, int np, uint32_t* u) {
    memset(u, 0, 64 * sizeof(uint32_t));
    for (int j = 0; j < np; j++) {
        uint64_t x = planes[j];
        uint32_t bit = 1u << (31 - j);
        while (x) {
            u[__builtin_ctzll(x)] |= bit;
            x &= x - 1;
        }
    }
}

#ifdef SZFP_X86
__attribute__((target("avx512f"))) static void
transpose_avx512(const uint64_t* planes, int np, uint32_t* u) {
    __m512i u0 = _mm512_setzero_si512();
    __m512i u1 = u0;
    __m512i u2 = u0;
    __m512i u3 = u0;

    for (int j = 0; j < np; j++) {
        uint64_t x = planes[j];
        __m512i bit = _mm512_set1_epi32((int)(1u << (31 - j)));
        u0 = _mm512_mask_or_epi32(u0, (__mmask16)x, u0, bit);
        u1 = _mm512_mask_or_epi32(u1, (__mmask16)(x >> 16), u1, bit);
        u2 = _mm512_mask_or_epi32(u2, (__mmask16)(x >> 32), u2, bit);
        u3 = _mm512_mask_or_epi32(u3, (__mmask16)(x >> 48), u3, bit);
    }
    _mm512_storeu_si512(u, u0);
    _mm512_storeu_si512(u + 16, u1);
    _mm512_storeu_si512(u + 32, u2);
    _mm512_storeu_si512(u + 48, u3);
}
#endif

typedef void (*TransposeFn)(const uint64_t*, int, uint32_t*);

static inline void inv_lift(int32_t* p, int s) {
    int32_t x = p[0];
    int32_t y = p[s];
    int32_t z = p[2 * s];
    int32_t w = p[3 * s];

    y += w >> 1;
    w -= y >> 1;
    y += w;
    w <<= 1;
    w -= y;
    z += x;
    x <<= 1;
    x -= z;
    y += z;
    z <<= 1;
    z -= y;
    w += x;
    x <<= 1;
    x -= w;

    p[0] = x;
    p[s] = y;
    p[2 * s] = z;
    p[3 * s] = w;
}

/* from the transposed coefficients u to the 64 values: negabinary to two's
 * complement, zfp's coefficient order, the inverse transform along z, y and x,
 * and the scaling by 2^(emax - 30) */
static void tail_scalar(const uint32_t* u, int emax, float* out) {
    int32_t iblock[64];
    float s;

    for (int i = 0; i < 64; i++) {
        iblock[PERM[i]] = (int32_t)((u[i] ^ NBMASK) - NBMASK);
    }

    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            inv_lift(iblock + x + 4 * y, 16);
        }
    }
    for (int x = 0; x < 4; x++) {
        for (int z = 0; z < 4; z++) {
            inv_lift(iblock + 16 * z + x, 4);
        }
    }
    for (int z = 0; z < 4; z++) {
        for (int y = 0; y < 4; y++) {
            inv_lift(iblock + 4 * y + 16 * z, 1);
        }
    }

    s = ldexpf(1.0f, emax - 30);
    for (int i = 0; i < 64; i++) {
        out[i] = s * (float)iblock[i];
    }
}

#ifdef SZFP_X86
/* coefficient index of each value, the inverse of PERM, for the gathers */
static int32_t INV[64];

/* inv_lift on 16 independent lines at once, one per lane */
__attribute__((target("avx512f"))) static inline void
inv_lift_avx512(__m512i* px, __m512i* py, __m512i* pz, __m512i* pw) {
    __m512i x = *px;
    __m512i y = *py;
    __m512i z = *pz;
    __m512i w = *pw;

    y = _mm512_add_epi32(y, _mm512_srai_epi32(w, 1));
    w = _mm512_sub_epi32(w, _mm512_srai_epi32(y, 1));
    y = _mm512_add_epi32(y, w);
    w = _mm512_slli_epi32(w, 1);
    w = _mm512_sub_epi32(w, y);
    z = _mm512_add_epi32(z, x);
    x = _mm512_slli_epi32(x, 1);
    x = _mm512_sub_epi32(x, z);
    y = _mm512_add_epi32(y, z);
    z = _mm512_slli_epi32(z, 1);
    z = _mm512_sub_epi32(z, y);
    w = _mm512_add_epi32(w, x);
    x = _mm512_slli_epi32(x, 1);
    x = _mm512_sub_epi32(x, w);

    *px = x;
    *py = y;
    *pz = z;
    *pw = w;
}

/* transpose the four 128-bit lanes across four vectors; its own inverse */
__attribute__((target("avx512f"))) static inline void
swap_lanes(__m512i* a, __m512i* b, __m512i* c, __m512i* d) {
    __m512i t0 = _mm512_shuffle_i32x4(*a, *b, 0x44);
    __m512i t1 = _mm512_shuffle_i32x4(*a, *b, 0xEE);
    __m512i t2 = _mm512_shuffle_i32x4(*c, *d, 0x44);
    __m512i t3 = _mm512_shuffle_i32x4(*c, *d, 0xEE);

    *a = _mm512_shuffle_i32x4(t0, t2, 0x88);
    *b = _mm512_shuffle_i32x4(t0, t2, 0xDD);
    *c = _mm512_shuffle_i32x4(t1, t3, 0x88);
    *d = _mm512_shuffle_i32x4(t1, t3, 0xDD);
}

/* transpose the four 32-bit values inside every 128-bit lane; its own inverse */
__attribute__((target("avx512f"))) static inline void
swap_values(__m512i* a, __m512i* b, __m512i* c, __m512i* d) {
    __m512i p = _mm512_unpacklo_epi32(*a, *b);
    __m512i q = _mm512_unpackhi_epi32(*a, *b);
    __m512i r = _mm512_unpacklo_epi32(*c, *d);
    __m512i t = _mm512_unpackhi_epi32(*c, *d);

    *a = _mm512_unpacklo_epi64(p, r);
    *b = _mm512_unpackhi_epi64(p, r);
    *c = _mm512_unpacklo_epi64(q, t);
    *d = _mm512_unpackhi_epi64(q, t);
}

/* tail_scalar on vectors: v[z] holds the 16 values x + 4y of plane z, so the z
 * pass works on whole vectors; the y and x passes do too after swapping lanes and
 * values, which are swapped back before the scaling */
__attribute__((target("avx512f"))) static void
tail_avx512(const uint32_t* u, int emax, float* out) {
    const __m512i nb = _mm512_set1_epi32((int)NBMASK);
    __m512i v[4];
    __m512 s;

    for (int z = 0; z < 4; z++) {
        __m512i idx = _mm512_loadu_si512(INV + 16 * z);
        __m512i c = _mm512_i32gather_epi32(idx, (const void*)u, 4);
        v[z] = _mm512_sub_epi32(_mm512_xor_si512(c, nb), nb);
    }

    inv_lift_avx512(&v[0], &v[1], &v[2], &v[3]); /* along z */
    swap_lanes(&v[0], &v[1], &v[2], &v[3]);      /* v[y], lanes x + 4z */
    inv_lift_avx512(&v[0], &v[1], &v[2], &v[3]); /* along y */
    swap_values(&v[0], &v[1], &v[2], &v[3]);     /* v[x], lanes y + 4z */
    inv_lift_avx512(&v[0], &v[1], &v[2], &v[3]); /* along x */
    swap_values(&v[0], &v[1], &v[2], &v[3]);
    swap_lanes(&v[0], &v[1], &v[2], &v[3]); /* v[z] again */

    s = _mm512_set1_ps(ldexpf(1.0f, emax - 30));
    for (int z = 0; z < 4; z++) {
        _mm512_storeu_ps(out + 16 * z, _mm512_mul_ps(s, _mm512_cvtepi32_ps(v[z])));
    }
}

/* tail_avx512 that counts lo < v <= hi instead of storing the values; the order
 * of the values does not matter for that, so they are not swapped back */
__attribute__((target("avx512f"))) static size_t
tail_count_avx512(const uint32_t* u, int emax, float lo_f, float hi_f) {
    const __m512i nb = _mm512_set1_epi32((int)NBMASK);
    const __m512 lo = _mm512_set1_ps(lo_f);
    const __m512 hi = _mm512_set1_ps(hi_f);
    __m512i v[4];
    __m512 s = _mm512_set1_ps(ldexpf(1.0f, emax - 30));
    size_t count = 0;

    for (int z = 0; z < 4; z++) {
        __m512i idx = _mm512_loadu_si512(INV + 16 * z);
        __m512i c = _mm512_i32gather_epi32(idx, (const void*)u, 4);
        v[z] = _mm512_sub_epi32(_mm512_xor_si512(c, nb), nb);
    }

    inv_lift_avx512(&v[0], &v[1], &v[2], &v[3]);
    swap_lanes(&v[0], &v[1], &v[2], &v[3]);
    inv_lift_avx512(&v[0], &v[1], &v[2], &v[3]);
    swap_values(&v[0], &v[1], &v[2], &v[3]);
    inv_lift_avx512(&v[0], &v[1], &v[2], &v[3]);

    for (int z = 0; z < 4; z++) {
        __m512 x = _mm512_mul_ps(s, _mm512_cvtepi32_ps(v[z]));
        __mmask16 in = _mm512_cmp_ps_mask(x, lo, _CMP_GT_OQ) &
                       _mm512_cmp_ps_mask(x, hi, _CMP_LE_OQ);
        count += (size_t)__builtin_popcount((unsigned)in);
    }
    return count;
}
#endif

typedef void (*TailFn)(const uint32_t*, int, float*);
typedef int (*PlanesFn)(const unsigned char*, uint32_t, uint32_t, uint64_t*);

static TransposeFn transpose_best = transpose_scalar;
static TransposeFn transpose = transpose_scalar;
static TailFn tail_best = tail_scalar;
static TailFn tail = tail_scalar;
static PlanesFn planes_best = decode_planes;
static PlanesFn planes = decode_planes;
static int count_best = 0; /* whether tail_count_avx512 can run */
static int count_fused = 0;

/* pick the SIMD steps once at load time so decoding threads never race on them */
__attribute__((constructor)) static void pick_simd(void) {
#ifdef SZFP_X86
    for (int i = 0; i < 64; i++) {
        INV[PERM[i]] = i;
    }
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f")) {
        transpose_best = transpose_avx512;
        tail_best = tail_avx512;
        count_best = 1;
    }
    if (__builtin_cpu_supports("bmi2") && __builtin_cpu_supports("popcnt")) {
        planes_best = decode_planes_bmi2;
    }
#endif
    transpose = transpose_best;
    tail = tail_best;
    planes = planes_best;
    count_fused = count_best;
}

static int fast_enabled = 1;

void szfp_set_fast_decode(int mode) {
    fast_enabled = mode != 0;
    transpose = mode == 2 ? transpose_scalar : transpose_best;
    tail = mode == 2 || mode == 3 ? tail_scalar : tail_best;
    planes = mode == 1 || mode == 5 ? planes_best : decode_planes;
    count_fused = mode == 1 ? count_best : 0;
}

int szfp_fast_count_ok(void) {
    return fast_enabled && count_fused;
}

int szfp_fast_ok(int block_dim, double rate, size_t block_nbytes) {
    return fast_enabled && block_dim == 4 && block_nbytes >= 2 &&
           block_nbytes <= SZFP_FAST_MAX_BYTES &&
           64.0 * rate == 8.0 * (double)block_nbytes;
}

void szfp_fast_decode(const unsigned char* src, size_t block_nbytes, float* out) {
    unsigned char buf[SZFP_FAST_MAX_BYTES + 8];
    uint64_t bit_planes[32];
    uint32_t u[64];
    int np;
    int emax;

    memcpy(buf, src, block_nbytes);
    memset(buf + block_nbytes, 0, 8);

    if (!(buf[0] & 1)) {
        memset(out, 0, 64 * sizeof(float));
        return;
    }

    emax = (int)((peek(buf, 1)) & 0xff) - 127;
    np = planes(buf, 9, (uint32_t)(8 * block_nbytes) - 9, bit_planes);
    transpose(bit_planes, np, u);
    tail(u, emax, out);
}

size_t
szfp_fast_count(const unsigned char* src, size_t block_nbytes, float lo, float hi) {
#ifdef SZFP_X86
    unsigned char buf[SZFP_FAST_MAX_BYTES + 8];
    uint64_t bit_planes[32];
    uint32_t u[64];
    int np;
    int emax;

    memcpy(buf, src, block_nbytes);
    memset(buf + block_nbytes, 0, 8);

    if (!(buf[0] & 1)) { /* an all-zero block */
        return (0.0f > lo && 0.0f <= hi) ? 64 : 0;
    }

    emax = (int)((peek(buf, 1)) & 0xff) - 127;
    np = planes(buf, 9, (uint32_t)(8 * block_nbytes) - 9, bit_planes);
    transpose(bit_planes, np, u);
    return tail_count_avx512(u, emax, lo, hi);
#else
    (void)src;
    (void)block_nbytes;
    (void)lo;
    (void)hi;
    return 0;
#endif
}
