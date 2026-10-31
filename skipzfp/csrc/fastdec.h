#ifndef SKIPZFP_FASTDEC_H
#define SKIPZFP_FASTDEC_H

#include <stddef.h>

/* Specialized decoder for fixed-rate 3D float blocks (4x4x4), bit-exact with
 * zfp_decode_block_float_3 built with the default ZFP_ROUND_NEVER. */

enum { SZFP_FAST_MAX_BYTES = 256 };

/**
 * @brief Choose how blocks are decoded, mainly for tests and benchmarks.
 *
 * @param[in] mode 1 (default) uses the fast path with every speedup the CPU
 *                 supports, 0 sends every block through libzfp, 2 uses the fast
 *                 path with no SIMD at all, 3 with the SIMD bit-plane transpose
 *                 only, 4 adds the SIMD inverse transform, and 5 also the BMI2
 *                 bit-plane decoder; 1 adds to 5 counting in SIMD registers.
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

/**
 * @brief Return nonzero if szfp_fast_count() can be used, which needs AVX-512.
 */
int szfp_fast_count_ok(void);

/**
 * @brief Count the values lo < v <= hi of one block without storing them.
 *
 * The bounds are floats: for float values, v > lo for a double lo is the same as
 * v > the largest float not above lo, and likewise for v <= hi, so callers round
 * their double bounds down to float first. The caller must check
 * szfp_fast_ok() and szfp_fast_count_ok() first.
 */
size_t szfp_fast_count(const unsigned char* src, size_t block_nbytes, float lo, float hi);

#endif
