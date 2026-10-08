#ifndef SKIPZFP_UTIL_H
#define SKIPZFP_UTIL_H

#include <stddef.h>
#include <stdint.h>
#include <zfp.h>

/** Result codes returned by every C function in the library (core.h, query.h). */
typedef enum {
    SZ_OK = 0,
    SZ_ERR_NULL = 1,       /**< a required pointer is NULL */
    SZ_ERR_STREAM = 2,     /**< ZFP could not open a bit stream */
    SZ_ERR_ZFP = 3,        /**< ZFP could not open a compression stream */
    SZ_ERR_FIELD = 4,      /**< ZFP could not create a field for the array */
    SZ_ERR_DECOMPRESS = 5, /**< ZFP decompression failed */
    SZ_ERR_DIMS = 6,       /**< unsupported number of dimensions */
    SZ_ERR_ARG = 7,        /**< invalid shape, rate or block size */
    SZ_ERR_SIZE = 8,       /**< a buffer has the wrong size or a size overflows */
    SZ_ERR_COMPRESS = 9,   /**< ZFP compression failed */
    SZ_ERR_MALLOC = 10     /**< out of memory */
} SzResult;

/* overflow-safe size arithmetic: each returns 0 on overflow, and out must not be NULL
 */
static inline int mul_size(size_t a, size_t b, size_t* out) {
    if (a != 0 && b > SIZE_MAX / a) {
        return 0;
    }

    *out = a * b;
    return 1;
}

static inline int add_size(size_t a, size_t b, size_t* out) {
    if (a > SIZE_MAX - b) {
        return 0;
    }

    *out = a + b;
    return 1;
}

/** Sizes of a chunk of nx * ny * nz values in 4x4x4 blocks at a fixed rate. */
typedef struct {
    size_t nval;
    size_t bx;
    size_t by;
    size_t bz;
    size_t nblk;
    size_t data_size;
    size_t meta_size;
} SzfpLayout;

/** @brief Fill lt for a chunk; return 0 if the shape, rate or block_dim
 *         (which must be 4) is invalid. */
int szfp_calc_layout(
    size_t nx,
    size_t ny,
    size_t nz,
    double rate,
    int block_dim,
    SzfpLayout* lt
);

/** @brief Return the C-order index of (x, y, z) in an array of shape (*, ny, nz). */
size_t get_index3(size_t x, size_t y, size_t z, size_t ny, size_t nz);

/** @brief Clamp x to 0..255 and return it as a byte. */
unsigned char to_u8(int x);

/** @brief Return a ZFP field for one block of 4^dims values in out, or NULL
 *         if dims is not 1 to 4. */
zfp_field* create_block_field(void* out, zfp_type type, int dims);

#endif
