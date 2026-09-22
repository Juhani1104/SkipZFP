#ifndef SKIPZFP_QUERY_H
#define SKIPZFP_QUERY_H

#include <stddef.h>
#include <stdint.h>

int szfp_plan_gt(
    const unsigned char* meta,
    size_t n_chunk,
    size_t meta_size,
    size_t blocks_per_chunk,
    double threshold,
    int threads,
    uint32_t* maybe_chunks,
    uint32_t* maybe_blocks,
    size_t maybe_cap,
    size_t* out_maybe,
    size_t* out_in,
    size_t* out_out
);

int szfp_count_gt_blocks(
    const unsigned char* blocks,
    size_t block_count,
    size_t block_nbytes,
    double rate,
    int block_dim,
    double threshold,
    int threads,
    size_t* out_count
);

int szfp_decode_blocks(
    const unsigned char* blocks,
    size_t block_count,
    size_t block_nbytes,
    double rate,
    int block_dim,
    int threads,
    float* out
);

#endif