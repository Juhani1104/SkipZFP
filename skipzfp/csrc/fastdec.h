#ifndef SKIPZFP_FASTDEC_H
#define SKIPZFP_FASTDEC_H

#include <stddef.h>

/* Specialized decoder for fixed-rate 3D float blocks (4x4x4), bit-exact with
 * zfp_decode_block_float_3 built with the default ZFP_ROUND_NEVER. */

enum { SZFP_FAST_MAX_BYTES = 256 };

/**
 * @brief Choose how blocks are decoded, mainly for tests and benchmarks.
 *
 * @param[in] mode 1 (default) uses the fast path, 0 sends every block through
 *                 libzfp, and 2 uses the fast path with the portable
 *                 (non-SIMD) bit-plane transpose.
 */
void szfp_set_fast_decode(int mode);

/**
 * @brief Return nonzero if szfp_fast_decode() can handle blocks of this
 *        layout.
 *
 * That requires the fast path to be on, block_dim to be 4, and block_nbytes
 * to match the rate and be at most SZFP_FAST_MAX_BYTES.
 */
int szfp_fast_ok(int block_dim, double rate, size_t block_nbytes);

/**
 * @brief Decode one 4x4x4 float block without going through libzfp.
 *
 * The caller must check szfp_fast_ok() first, since this function does no
 * checks of its own.
 *
 * @param[in]  src          Compressed bytes of the block.
 * @param[in]  block_nbytes Size of src.
 * @param[out] out          Buffer that receives the 64 values.
 */
void szfp_fast_decode(const unsigned char* src, size_t block_nbytes, float* out);

#endif
