#ifndef SKIPZFP_UTIL_H
#define SKIPZFP_UTIL_H

#include <stddef.h>
#include <zfp.h>

/** @brief Return the C-order index of (x, y, z) in an array of shape (*, ny, nz). */
size_t get_index3(size_t x, size_t y, size_t z, size_t ny, size_t nz);

/** @brief Clamp x to 0..255 and return it as a byte. */
unsigned char to_u8(int x);

/** @brief Return a ZFP field for one block of 4^dims values in out, or NULL
 *         if dims is not 1 to 4. */
zfp_field* create_block_field(void* out, zfp_type type, int dims);

#endif
