#ifndef SKIPZFP_CORE_H
#define SKIPZFP_CORE_H

#include <stddef.h>

#include "util.h"

/**
 * @brief Compute the compressed size of a chunk and the size of its zone map
 *        record.
 *
 * @param[in]  nx, ny, nz Chunk shape in C order, each a multiple of block_dim.
 * @param[in]  rate       ZFP fixed rate in bits per value.
 * @param[in]  block_dim  Edge length of a ZFP block.
 * @param[out] data_size  Bytes of compressed data, nx * ny * nz * rate / 8.
 * @param[out] meta_size  Bytes of the zone map record, which is 12 bytes plus
 *                        2 per block.
 *
 * @return SZ_OK on success, SZ_ERR_NULL if an output pointer is NULL, or
 *         SZ_ERR_ARG if the shape, rate or block_dim is invalid.
 */
SzResult szfp_layout(
    size_t nx,
    size_t ny,
    size_t nz,
    double rate,
    int block_dim,
    size_t* data_size,
    size_t* meta_size
);

/**
 * @brief Compress a float32 chunk with fixed-rate ZFP.
 *
 * Every 4x4x4 block takes the same number of bytes, and blocks are written in
 * C order, so block i starts at byte i * 64 * rate / 8.
 *
 * @param[in]  chunk        The nx * ny * nz values of the chunk in C order.
 * @param[in]  nx, ny, nz   Chunk shape, each a multiple of 4.
 * @param[in]  rate         ZFP fixed rate in bits per value.
 * @param[in]  block_dim    Edge length of a ZFP block, which must be 4.
 * @param[out] out          Buffer that receives the compressed bytes.
 * @param[in]  out_capacity Size of out, which must be at least the data_size
 *                          from szfp_layout().
 * @param[out] out_size     Number of bytes written.
 *
 * @return SZ_OK on success, SZ_ERR_NULL if a pointer is NULL, SZ_ERR_ARG if
 *         the shape, rate or block_dim is invalid, SZ_ERR_SIZE if out is too
 *         small, or a ZFP error code if compression fails.
 */
SzResult szfp_encode(
    const float* chunk,
    size_t nx,
    size_t ny,
    size_t nz,
    double rate,
    int block_dim,
    unsigned char* out,
    size_t out_capacity,
    size_t* out_size
);

/**
 * @brief Decompress a chunk written by szfp_encode().
 *
 * @param[in]  src        Compressed bytes of the whole chunk.
 * @param[in]  src_size   Size of src, which must equal the data_size from
 *                        szfp_layout().
 * @param[in]  nx, ny, nz Chunk shape in C order.
 * @param[in]  rate       ZFP fixed rate the chunk was written with.
 * @param[in]  block_dim  Edge length of a ZFP block.
 * @param[out] out        Buffer that receives nx * ny * nz values in C order.
 * @param[in]  out_count  Length of out in values.
 *
 * @return SZ_OK on success, SZ_ERR_NULL if a pointer is NULL, SZ_ERR_ARG if
 *         the shape, rate or block_dim is invalid, SZ_ERR_SIZE if src or out
 *         has the wrong size, or a ZFP error code if decompression fails.
 */
SzResult szfp_decode(
    const unsigned char* src,
    size_t src_size,
    size_t nx,
    size_t ny,
    size_t nz,
    double rate,
    int block_dim,
    float* out,
    size_t out_count
);

/**
 * @brief Compute the zone map record of a chunk at one rate.
 *
 * The record holds the chunk's min and max and the largest ZFP error at this
 * rate as three float32, followed by two uint8 per block in C order, which
 * place the block's min and max on a 0 to 255 scale between the chunk's min
 * and max, rounded outward so the true range always fits inside.
 *
 * @param[in]  chunk         Original values in C order.
 * @param[in]  nx, ny, nz    Chunk shape, each a multiple of 4.
 * @param[in]  rate          ZFP fixed rate used to measure the error.
 * @param[in]  block_dim     Edge length of a ZFP block, which must be 4.
 * @param[out] meta          Buffer that receives the record.
 * @param[in]  meta_capacity Size of meta, which must be at least the
 *                           meta_size from szfp_layout().
 *
 * @return SZ_OK on success, SZ_ERR_NULL if a pointer is NULL, SZ_ERR_ARG if
 *         the shape, rate or block_dim is invalid, SZ_ERR_SIZE if meta is too
 *         small, SZ_ERR_MALLOC if scratch memory runs out, or a ZFP error code
 *         if compressing or decompressing the chunk fails.
 */
SzResult szfp_meta(
    const float* chunk,
    size_t nx,
    size_t ny,
    size_t nz,
    double rate,
    int block_dim,
    unsigned char* meta,
    size_t meta_capacity
);

/**
 * @brief Decompress a single fixed-rate ZFP block of 1 to 4 dimensions.
 *
 * @param[in]  block_bytes    Compressed bytes of the block.
 * @param[in]  block_nbytes   Size of block_bytes.
 * @param[in]  rate           ZFP fixed rate in bits per value.
 * @param[in]  zfp_type_value A zfp_type value, such as zfp_type_float.
 * @param[in]  dims           Number of dimensions, from 1 to 4.
 * @param[out] out            Buffer that receives the 4^dims values.
 *
 * @return SZ_OK on success, SZ_ERR_NULL if a pointer is NULL, SZ_ERR_ARG if
 *         the rate or size is invalid, SZ_ERR_DIMS if dims is out of range,
 *         SZ_ERR_SIZE if block_bytes is too short for the rate, or a ZFP error
 *         code if decompression fails.
 */
SzResult szfp_decode_block(
    const unsigned char* block_bytes,
    size_t block_nbytes,
    double rate,
    int zfp_type_value,
    int dims,
    void* out
);

#endif
