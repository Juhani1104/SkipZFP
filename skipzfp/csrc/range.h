#ifndef SKIPZFP_RANGE_H
#define SKIPZFP_RANGE_H

#include <stddef.h>
#include <stdint.h>

/*
 * These are the range versions of the functions in query.h, kept apart until
 * the two are merged. They test lo < x <= hi instead of x > threshold, where either
 * bound may be infinite, and return the same codes as their query.h counterparts.
 */

/**
 * @brief Classify every block for lo < x <= hi and list the MAYBE blocks in
 *        chunk order.
 *
 * Same as szfp_plan_gt_chunks() with threshold replaced by lo and hi.
 */
int szfp_plan_range_chunks(
    const unsigned char* meta,
    size_t meta_size,
    size_t ux,
    size_t uy,
    size_t uz,
    size_t sx,
    size_t sy,
    size_t sz,
    size_t blocks_per_unit,
    double lo,
    double hi,
    int threads,
    uint32_t* maybe_chunks,
    uint32_t* maybe_blocks,
    size_t maybe_cap,
    size_t* out_maybe,
    size_t* out_in,
    size_t* out_out
);

/**
 * @brief Decode packed blocks and count the values with lo < x <= hi.
 *
 * Same as szfp_count_gt_blocks() with threshold replaced by lo and hi.
 */
int szfp_count_range_blocks(
    const unsigned char* blocks,
    size_t block_count,
    size_t block_nbytes,
    double rate,
    int block_dim,
    double lo,
    double hi,
    int threads,
    size_t* out_count
);

/**
 * @brief Decode the blocks at the given offsets in a buffer and count the
 *        values with lo < x <= hi.
 *
 * Same as szfp_count_offsets() with threshold replaced by lo and hi.
 */
int szfp_count_offsets_range(
    const unsigned char* buf,
    size_t buf_size,
    const uint64_t* offsets,
    size_t n,
    size_t block_nbytes,
    double rate,
    int block_dim,
    double lo,
    double hi,
    size_t* out_count
);

#endif
