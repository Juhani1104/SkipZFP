#ifndef SKIPZFP_FASTDEC_H
#define SKIPZFP_FASTDEC_H

#include <stddef.h>

/* Specialized decoder for fixed-rate 3D float blocks (4x4x4), bit-exact with
 * zfp_decode_block_float_3 built with the default ZFP_ROUND_NEVER. */

enum { SZFP_FAST_MAX_BYTES = 256 };

/* 1 (default): fast path; 0: every block through libzfp;
 * 2: fast path with the portable (non-SIMD) bit-plane transpose */
void szfp_set_fast_decode(int mode);

int szfp_fast_ok(int block_dim, double rate, size_t block_nbytes);

void szfp_fast_decode(const unsigned char* src, size_t block_nbytes, float* out);

#endif
