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

static TransposeFn transpose_best = transpose_scalar;
static TransposeFn transpose = transpose_scalar;

/* pick the transpose once at load time so decoding threads never race on it */
__attribute__((constructor)) static void pick_transpose(void) {
#ifdef SZFP_X86
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f")) {
        transpose_best = transpose_avx512;
    }
#endif
    transpose = transpose_best;
}

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

static int fast_enabled = 1;

void szfp_set_fast_decode(int mode) {
    fast_enabled = mode != 0;
    transpose = mode == 2 ? transpose_scalar : transpose_best;
}

int szfp_fast_ok(int block_dim, double rate, size_t block_nbytes) {
    return fast_enabled && block_dim == 4 && block_nbytes >= 2 &&
           block_nbytes <= SZFP_FAST_MAX_BYTES &&
           64.0 * rate == 8.0 * (double)block_nbytes;
}

void szfp_fast_decode(const unsigned char* src, size_t block_nbytes, float* out) {
    unsigned char buf[SZFP_FAST_MAX_BYTES + 8];
    uint64_t planes[32];
    uint32_t u[64];
    int32_t iblock[64];
    int np;
    int emax;
    float s;

    memcpy(buf, src, block_nbytes);
    memset(buf + block_nbytes, 0, 8);

    if (!(buf[0] & 1)) {
        memset(out, 0, 64 * sizeof(float));
        return;
    }

    emax = (int)((peek(buf, 1)) & 0xff) - 127;
    np = decode_planes(buf, 9, (uint32_t)(8 * block_nbytes) - 9, planes);
    transpose(planes, np, u);

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
