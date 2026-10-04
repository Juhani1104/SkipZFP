#ifndef SKIPZFP_QUERY_H
#define SKIPZFP_QUERY_H

#include <stddef.h>
#include <stdint.h>

/*
 * Each x > threshold function is the range version with hi = +infinity.
 *
 * The functions here return 0 on success or one of these codes, which are
 * defined in query.c:
 *   1  a required pointer is NULL
 *   2  invalid argument, such as mismatched sizes or unsorted input
 *   3  a buffer has the wrong size or a size overflows
 *   4  out of memory
 *   5  ZFP decoding failed
 */

/**
 * @brief Classify every block for x > threshold and list the MAYBE blocks in
 *        chunk order.
 *
 * For every block, the min and max from the zone map are widened by the error
 * bound of its unit and then compared with the threshold. The MAYBE blocks
 * come out sorted by chunk and then by position in the chunk, so they can be
 * merged into reads without another sort.
 *
 * @param[in]  meta            Zone map of all units in C order, with one
 *                             record of meta_size bytes per unit.
 * @param[in]  meta_size       Bytes per record, which must be
 *                             12 + 2 * blocks_per_unit.
 * @param[in]  ux, uy, uz      Units along each axis of the whole array.
 * @param[in]  sx, sy, sz      Units along each axis of one chunk.
 * @param[in]  blocks_per_unit Blocks in one unit.
 * @param[in]  threshold       Count values strictly greater than this.
 * @param[in]  threads         OpenMP threads, where <= 0 uses all cores.
 * @param[out] maybe_chunks    Chunk id of each MAYBE block.
 * @param[out] maybe_blocks    Position of each MAYBE block in its chunk, in
 *                             storage order.
 * @param[in]  maybe_cap       Length of both output arrays, which must be at
 *                             least the total number of blocks.
 * @param[out] out_maybe       Number of MAYBE blocks written.
 * @param[out] out_in          Number of IN blocks.
 * @param[out] out_out         Number of OUT blocks.
 *
 * @return 0 on success, 1 if a pointer is NULL, 2 if the sizes do not match
 *         or maybe_cap is too small, or 4 if scratch memory runs out.
 */
int szfp_plan_gt_chunks(
    const unsigned char* meta,
    size_t meta_size,
    size_t ux,
    size_t uy,
    size_t uz,
    size_t sx,
    size_t sy,
    size_t sz,
    size_t blocks_per_unit,
    double threshold,
    int threads,
    uint32_t* maybe_chunks,
    uint32_t* maybe_blocks,
    size_t maybe_cap,
    size_t* out_maybe,
    size_t* out_in,
    size_t* out_out
);

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
 * @brief Classify every block for x > threshold with the older unit-order
 *        planner.
 *
 * It works like szfp_plan_gt_chunks(), but treats each record as its own
 * chunk, so the MAYBE blocks come out in unit order. The Python side then
 * remaps them to chunk order.
 *
 * @param[in]  meta             Zone map of all units, one record each.
 * @param[in]  n_chunk          Number of records.
 * @param[in]  meta_size        Bytes per record, which must be
 *                              12 + 2 * blocks_per_chunk.
 * @param[in]  blocks_per_chunk Blocks covered by one record.
 * @param[in]  threshold        Count values strictly greater than this.
 * @param[in]  threads          OpenMP threads, where <= 0 uses all cores.
 * @param[out] maybe_chunks     Record id of each MAYBE block.
 * @param[out] maybe_blocks     Position of each MAYBE block in its record.
 * @param[in]  maybe_cap        Length of both output arrays, which must be at
 *                              least n_chunk * blocks_per_chunk.
 * @param[out] out_maybe        Number of MAYBE blocks written.
 * @param[out] out_in           Number of IN blocks.
 * @param[out] out_out          Number of OUT blocks.
 *
 * @return 0 on success, 1 if a pointer is NULL, 2 if the sizes do not match
 *         or maybe_cap is too small, 3 if a size overflows, or 4 if scratch
 *         memory runs out.
 */
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

/**
 * @brief Decode packed blocks and count the values greater than threshold.
 *
 * @param[in]  blocks       The block_count blocks, stored back to back.
 * @param[in]  block_count  Number of blocks.
 * @param[in]  block_nbytes Bytes per block, which must equal
 *                          block_dim^3 * rate / 8.
 * @param[in]  rate         ZFP fixed rate in bits per value.
 * @param[in]  block_dim    Edge length of a ZFP block.
 * @param[in]  threshold    Count values strictly greater than this.
 * @param[in]  threads      OpenMP threads, where <= 0 uses all cores.
 * @param[out] out_count    Number of matching values.
 *
 * @return 0 on success, 1 if a pointer is NULL, 2 if rate or block_dim is
 *         invalid, 3 if block_nbytes does not match, 4 if scratch memory runs
 *         out, or 5 if a block fails to decode.
 */
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
 * @brief Decode packed blocks into their values.
 *
 * @param[in]  blocks       The block_count blocks, stored back to back.
 * @param[in]  block_count  Number of blocks.
 * @param[in]  block_nbytes Bytes per block, which must equal
 *                          block_dim^3 * rate / 8.
 * @param[in]  rate         ZFP fixed rate in bits per value.
 * @param[in]  block_dim    Edge length of a ZFP block.
 * @param[in]  threads      OpenMP threads, where <= 0 uses all cores.
 * @param[out] out          Buffer that receives block_dim^3 values per block,
 *                          block after block.
 *
 * @return 0 on success, 1 if a pointer is NULL, 2 if rate or block_dim is
 *         invalid, 3 if block_nbytes does not match, or 5 if a block fails to
 *         decode.
 */
int szfp_decode_blocks(
    const unsigned char* blocks,
    size_t block_count,
    size_t block_nbytes,
    double rate,
    int block_dim,
    int threads,
    float* out
);

/**
 * @brief Group sorted (chunk, block) ids into runs that can share one read.
 *
 * Two ids join the same run when they are in the same chunk and at most
 * gap_blocks unneeded blocks lie between them. Each output array needs room
 * for n entries, which is the most runs there can be.
 *
 * @param[in]  chunk_ids      Chunk id of each block, sorted.
 * @param[in]  block_ids      Block id of each block, sorted within a chunk.
 * @param[in]  n              Number of ids.
 * @param[in]  gap_blocks     Largest gap that still joins two ids.
 * @param[out] out_chunk      Chunk of each run.
 * @param[out] out_first      First block of each run.
 * @param[out] out_last       Last block of each run.
 * @param[out] out_item_start Index in the input of each run's first id.
 * @param[out] out_n          Number of runs.
 *
 * @return 0 on success, 1 if a pointer is NULL, or 2 if the ids are not
 *         strictly increasing.
 */
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

/**
 * @brief Decode the blocks at the given offsets in a buffer and count the
 *        values greater than threshold.
 *
 * It runs on one thread, since the caller already decodes many buffers at
 * once while they are downloaded.
 *
 * @param[in]  buf          Bytes from one read.
 * @param[in]  buf_size     Size of buf.
 * @param[in]  offsets      Byte offset of each block to decode.
 * @param[in]  n            Number of offsets.
 * @param[in]  block_nbytes Bytes per block, which must equal 64 * rate / 8.
 * @param[in]  rate         ZFP fixed rate in bits per value.
 * @param[in]  block_dim    Edge length of a ZFP block, which must be 4.
 * @param[in]  threshold    Count values strictly greater than this.
 * @param[out] out_count    Number of matching values.
 *
 * @return 0 on success, 1 if a pointer is NULL, 2 if rate or block_dim is
 *         invalid, 3 if block_nbytes does not match or a block runs past the
 *         end of buf, or 5 if ZFP cannot open a stream.
 */
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
