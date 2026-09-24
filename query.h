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

int szfp_merge_ranges(
    const uint32_t* chunk_ids,
    const uint32_t* block_ids,
    size_t n,
    size_t gap_blocks,
    uint32_t* out_chunk,
    uint64_t* out_first,
    uint64_t* out_last,
    uint64_t* out_item_start,
    size_t* out_n
);

int szfp_count_offsets(
    const unsigned char* buf,
    size_t buf_size,
    const uint64_t* offsets,
    size_t n,
    size_t block_nbytes,
    double rate,
    int block_dim,
    double threshold,
    size_t* out_count
);

#endif