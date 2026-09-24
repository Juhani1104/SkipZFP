#ifndef SKIPZFP_CORE_H
#define SKIPZFP_CORE_H

#include <stddef.h>

typedef enum {
    SZ_OK = 0,
    SZ_ERR_NULL = 1,
    SZ_ERR_STREAM = 2,
    SZ_ERR_ZFP = 3,
    SZ_ERR_FIELD = 4,
    SZ_ERR_DECOMPRESS = 5,
    SZ_ERR_DIMS = 6,
    SZ_ERR_ARG = 7,
    SZ_ERR_SIZE = 8,
    SZ_ERR_COMPRESS = 9,
    SZ_ERR_MALLOC = 10
} SzResult;

SzResult szfp_layout(
    size_t nx,
    size_t ny,
    size_t nz,
    double rate,
    int block_dim,
    size_t* data_size,
    size_t* meta_size
);

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

SzResult szfp_decode_block(
    const unsigned char* block_bytes,
    size_t block_nbytes,
    double rate,
    int zfp_type_value,
    int dims,
    void* out
);

#endif