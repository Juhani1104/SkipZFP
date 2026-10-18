#include <zfp.h>

#include "fastdec.h"
#include "query.h"
#include "util.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* the value predicate of a range query: lo < v <= hi */
static inline int in_range(float v, double lo, double hi) {
    double d = (double)v;
    return (d > lo) & (d <= hi); /* & rather than && keeps the loop branch-free */
}

/* count values with lo < v <= hi; an open upper end takes the same single
 * comparison as the x > T path, so it costs the same */
static size_t count_in_range(const float* v, size_t n, double lo, double hi) {
    size_t c = 0;

    if (hi == INFINITY) {
        for (size_t i = 0; i < n; i++) {
            c += (double)v[i] > lo;
        }
    } else {
        for (size_t i = 0; i < n; i++) {
            c += in_range(v[i], lo, hi);
        }
    }

    return c;
}

static int block_layout(int block_dim, double rate, size_t* nval, size_t* nbytes) {
    size_t bd;
    size_t b2;
    size_t b3;

    if (nval == NULL || nbytes == NULL) {
        return 0;
    }

    if (block_dim <= 0 || rate <= 0.0 || !isfinite(rate)) {
        return 0;
    }

    bd = (size_t)block_dim;

    if (!mul_size(bd, bd, &b2) || !mul_size(b2, bd, &b3)) {
        return 0;
    }

    *nval = b3;
    *nbytes = (size_t)((double)b3 * rate / 8.0);

    if (*nbytes == 0) {
        return 0;
    }

    return 1;
}

static int decode_block(
    const unsigned char* src,
    size_t src_size,
    double rate,
    int block_dim,
    float* out
) {
    bitstream* stream;
    zfp_stream* zfp;
    zfp_field* field;
    size_t ret;

    if (src == NULL || out == NULL) {
        return SZ_ERR_NULL;
    }

    if (src_size == 0 || block_dim <= 0 || rate <= 0.0 || !isfinite(rate)) {
        return SZ_ERR_ARG;
    }

    if (szfp_fast_ok(block_dim, rate, src_size)) {
        szfp_fast_decode(src, src_size, out);
        return SZ_OK;
    }

    stream = stream_open((void*)src, src_size);
    if (stream == NULL) {
        return SZ_ERR_ZFP;
    }

    zfp = zfp_stream_open(stream);
    if (zfp == NULL) {
        stream_close(stream);
        return SZ_ERR_ZFP;
    }

    zfp_stream_set_rate(zfp, rate, zfp_type_float, 3, 0);
    zfp_stream_rewind(zfp);

    field = zfp_field_3d(
        out, zfp_type_float, (uint)block_dim, (uint)block_dim, (uint)block_dim
    );

    if (field == NULL) {
        zfp_stream_close(zfp);
        stream_close(stream);
        return SZ_ERR_ZFP;
    }

    ret = zfp_decompress(zfp, field);

    zfp_field_free(field);
    zfp_stream_close(zfp);
    stream_close(stream);

    return ret == 0 ? SZ_ERR_ZFP : SZ_OK;
}

/* bytes of per-block 4-bit error codes in a record of n_block blocks */
static size_t code_bytes(size_t n_block) {
    return (n_block + 1u) / 2u;
}

/* error bound of block bid: the unit's eps, or eps * q / 15 when the record
 * stores a 4-bit code q per block (low nibble first). The writer checks that
 * this very expression covers the block's error, so it must not change. */
static inline double block_error(float eps, const unsigned char* codes, size_t bid) {
    unsigned q;

    if (codes == NULL) {
        return (double)eps;
    }
    q = (codes[bid / 2u] >> (4u * (bid % 2u))) & 15u;
    return (double)eps * (double)q / 15.0;
}

/* classify the blocks of one metadata unit for lo < x <= hi:
 * 0 = OUT, 1 = IN, 2 = MAYBE. has_codes says whether the record ends with
 * per-block error codes. */
static void classify_unit(
    const unsigned char* cm,
    size_t n_block,
    int has_codes,
    double lo,
    double hi,
    unsigned char* states
) {
    const unsigned char* offs = cm + 3u * sizeof(float);
    const unsigned char* codes = has_codes ? offs + 2u * n_block : NULL;
    float cmin;
    float cmax;
    float eps;
    double span;
    double slack;

    memcpy(&cmin, cm, sizeof(float));
    memcpy(&cmax, cm + sizeof(float), sizeof(float));
    memcpy(&eps, cm + 2u * sizeof(float), sizeof(float));

    span = (double)cmax - (double)cmin;
    slack = 8.0 * DBL_EPSILON * (fabs((double)cmin) + fabs((double)cmax));

    for (size_t bid = 0; bid < n_block; bid++) {
        double bmin;
        double bmax;
        double e = block_error(eps, codes, bid);
        unsigned char state = 2;

        if (span == 0.0) {
            bmin = cmin;
            bmax = cmax;
        } else {
            bmin = (double)cmin + ((double)offs[bid * 2u] / 255.0) * span - slack;
            bmax = (double)cmin + ((double)offs[bid * 2u + 1u] / 255.0) * span + slack;
        }

        /* predicate lo < x <= hi; either bound may be infinite */
        if (bmin - e > lo && bmax + e <= hi) {
            state = 1;
        } else if (bmax + e <= lo || bmin - e > hi) {
            state = 0;
        }

        states[bid] = state;
    }
}

/* classify every block for lo < x <= hi and list the blocks a query reads,
 * sorted by (chunk, block): the MAYBE blocks, plus the IN blocks when keep_in
 * is set. When states is not NULL, it receives each listed block's state. */
static int plan_blocks(
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
    int keep_in,
    uint32_t* maybe_chunks,
    uint32_t* maybe_blocks,
    unsigned char* states_out,
    size_t maybe_cap,
    size_t* out_maybe,
    size_t* out_in,
    size_t* out_out
) {
    size_t gx;
    size_t gy;
    size_t gz;
    size_t n_chunk;
    size_t n_sub;
    size_t blocks_per_chunk;
    size_t total_blocks;
    unsigned char* states;
    size_t* n_maybe_chunk;
    size_t n_in = 0;
    size_t n_out = 0;
    int workers;
    int has_codes;

    if (meta == NULL || maybe_chunks == NULL || maybe_blocks == NULL ||
        out_maybe == NULL || out_in == NULL || out_out == NULL) {
        return SZ_ERR_NULL;
    }

    *out_maybe = 0;
    *out_in = 0;
    *out_out = 0;

    if (sx == 0 || sy == 0 || sz == 0 || ux % sx || uy % sy || uz % sz ||
        blocks_per_unit == 0 ||
        (meta_size != 3u * sizeof(float) + 2u * blocks_per_unit &&
         meta_size !=
             3u * sizeof(float) + 2u * blocks_per_unit + code_bytes(blocks_per_unit))) {
        return SZ_ERR_ARG;
    }

    has_codes = meta_size != 3u * sizeof(float) + 2u * blocks_per_unit;
    gx = ux / sx;
    gy = uy / sy;
    gz = uz / sz;
    n_sub = sx * sy * sz;
    n_chunk = gx * gy * gz;

    if (n_chunk == 0 || !mul_size(n_sub, blocks_per_unit, &blocks_per_chunk) ||
        !mul_size(n_chunk, blocks_per_chunk, &total_blocks) ||
        maybe_cap < total_blocks || blocks_per_chunk > UINT32_MAX ||
        n_chunk > UINT32_MAX) {
        return SZ_ERR_ARG;
    }

    states = (unsigned char*)malloc(total_blocks);
    n_maybe_chunk = (size_t*)malloc((n_chunk + 1u) * sizeof(size_t));
    if (states == NULL || n_maybe_chunk == NULL) {
        free(states);
        free(n_maybe_chunk);
        return SZ_ERR_MALLOC;
    }

    workers = threads > 0 ? threads : 1;
#ifdef _OPENMP
    if (threads <= 0) {
        workers = omp_get_max_threads();
    }
#endif

    /* pass 1: classify in chunk order and count per chunk */
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : n_in, n_out) schedule(static)                   \
    num_threads(workers)
#endif
    for (size_t cid = 0; cid < n_chunk; cid++) {
        size_t cx = cid / (gy * gz);
        size_t cy = (cid / gz) % gy;
        size_t cz = cid % gz;
        unsigned char* st = states + cid * blocks_per_chunk;
        size_t m = 0;

        for (size_t sub = 0; sub < n_sub; sub++) {
            size_t ix = cx * sx + sub / (sy * sz);
            size_t iy = cy * sy + (sub / sz) % sy;
            size_t iz = cz * sz + sub % sz;
            size_t uid = (ix * uy + iy) * uz + iz;
            unsigned char* su = st + sub * blocks_per_unit;

            classify_unit(
                meta + uid * meta_size, blocks_per_unit, has_codes, lo, hi, su
            );
        }

        for (size_t b = 0; b < blocks_per_chunk; b++) {
            n_in += st[b] == 1;
            n_out += st[b] == 0;
            m += st[b] == 2 || (keep_in && st[b] == 1);
        }
        n_maybe_chunk[cid + 1u] = m;
    }

    n_maybe_chunk[0] = 0;
    for (size_t cid = 0; cid < n_chunk; cid++) {
        n_maybe_chunk[cid + 1u] += n_maybe_chunk[cid];
    }

    /* pass 2: write the listed blocks, already sorted by (chunk, block) */
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(workers)
#endif
    for (size_t cid = 0; cid < n_chunk; cid++) {
        const unsigned char* st = states + cid * blocks_per_chunk;
        size_t k = n_maybe_chunk[cid];

        for (size_t b = 0; b < blocks_per_chunk; b++) {
            if (st[b] == 2 || (keep_in && st[b] == 1)) {
                maybe_chunks[k] = (uint32_t)cid;
                maybe_blocks[k] = (uint32_t)b;
                if (states_out != NULL) {
                    states_out[k] = st[b];
                }
                k++;
            }
        }
    }

    *out_maybe = n_maybe_chunk[n_chunk];
    *out_in = n_in;
    *out_out = n_out;

    free(states);
    free(n_maybe_chunk);
    return SZ_OK;
}

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
) {
    return plan_blocks(
        meta,
        meta_size,
        ux,
        uy,
        uz,
        sx,
        sy,
        sz,
        blocks_per_unit,
        lo,
        hi,
        threads,
        0,
        maybe_chunks,
        maybe_blocks,
        NULL,
        maybe_cap,
        out_maybe,
        out_in,
        out_out
    );
}

int szfp_plan_read_chunks(
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
    uint32_t* read_chunks,
    uint32_t* read_blocks,
    unsigned char* read_states,
    size_t read_cap,
    size_t* out_read,
    size_t* out_in,
    size_t* out_out
) {
    if (read_states == NULL) {
        return SZ_ERR_NULL;
    }

    return plan_blocks(
        meta,
        meta_size,
        ux,
        uy,
        uz,
        sx,
        sy,
        sz,
        blocks_per_unit,
        lo,
        hi,
        threads,
        1,
        read_chunks,
        read_blocks,
        read_states,
        read_cap,
        out_read,
        out_in,
        out_out
    );
}

/* x > threshold is the range threshold < x <= +inf */
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
) {
    return szfp_plan_range_chunks(
        meta,
        meta_size,
        ux,
        uy,
        uz,
        sx,
        sy,
        sz,
        blocks_per_unit,
        threshold,
        INFINITY,
        threads,
        maybe_chunks,
        maybe_blocks,
        maybe_cap,
        out_maybe,
        out_in,
        out_out
    );
}

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
) {
    size_t nval;
    size_t need_bytes;
    size_t scratch_vals;
    size_t scratch_bytes;
    size_t count;
    float* scratch;
    int failed;
    int workers;

    if (out_count == NULL) {
        return SZ_ERR_NULL;
    }

    *out_count = 0;

    if (block_count == 0) {
        return SZ_OK;
    }

    if (blocks == NULL) {
        return SZ_ERR_NULL;
    }

    if (!block_layout(block_dim, rate, &nval, &need_bytes)) {
        return SZ_ERR_ARG;
    }

    if (block_nbytes != need_bytes) {
        return SZ_ERR_SIZE;
    }

    workers = threads > 0 ? threads : 1;

#ifdef _OPENMP
    if (threads <= 0) {
        workers = omp_get_max_threads();
    }
#endif

    if (workers <= 0 || !mul_size((size_t)workers, nval, &scratch_vals) ||
        !mul_size(scratch_vals, sizeof(float), &scratch_bytes)) {
        return SZ_ERR_SIZE;
    }

    scratch = (float*)malloc(scratch_bytes);
    if (scratch == NULL) {
        return SZ_ERR_MALLOC;
    }

    count = 0;
    failed = 0;

#ifdef _OPENMP
#pragma omp parallel for reduction(+ : count) reduction(| : failed) schedule(static)   \
    num_threads(workers)
#endif
    for (size_t bid = 0; bid < block_count; bid++) {
        int tid = 0;
        float* vals;
        size_t local_count;
        int code;

#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif

        if (tid < 0 || tid >= workers) {
            failed = 1;
            continue;
        }

        vals = scratch + (size_t)tid * nval;

        code = decode_block(
            blocks + bid * block_nbytes, block_nbytes, rate, block_dim, vals
        );

        if (code != SZ_OK) {
            failed = 1;
            continue;
        }

        local_count = count_in_range(vals, nval, lo, hi);

        count += local_count;
    }

    free(scratch);

    if (failed) {
        return SZ_ERR_ZFP;
    }

    *out_count = count;
    return SZ_OK;
}

int szfp_count_gt_blocks(
    const unsigned char* blocks,
    size_t block_count,
    size_t block_nbytes,
    double rate,
    int block_dim,
    double threshold,
    int threads,
    size_t* out_count
) {
    return szfp_count_range_blocks(
        blocks,
        block_count,
        block_nbytes,
        rate,
        block_dim,
        threshold,
        INFINITY,
        threads,
        out_count
    );
}
int szfp_decode_blocks(
    const unsigned char* blocks,
    size_t block_count,
    size_t block_nbytes,
    double rate,
    int block_dim,
    int threads,
    float* out
) {
    size_t nval;
    size_t need_bytes;
    int failed;
    int workers;

    if (block_count == 0) {
        return SZ_OK;
    }

    if (blocks == NULL || out == NULL) {
        return SZ_ERR_NULL;
    }

    if (!block_layout(block_dim, rate, &nval, &need_bytes)) {
        return SZ_ERR_ARG;
    }

    if (block_nbytes != need_bytes) {
        return SZ_ERR_SIZE;
    }

    workers = threads > 0 ? threads : 1;
    failed = 0;

#ifdef _OPENMP
    if (threads <= 0) {
        workers = omp_get_max_threads();
    }
#pragma omp parallel for reduction(| : failed) schedule(static) num_threads(workers)
#endif
    for (size_t bid = 0; bid < block_count; bid++) {
        if (decode_block(
                blocks + bid * block_nbytes,
                block_nbytes,
                rate,
                block_dim,
                out + bid * nval
            ) != SZ_OK) {
            failed = 1;
        }
    }

    return failed ? SZ_ERR_ZFP : SZ_OK;
}

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
) {
    size_t m = 0;

    if (out_n == NULL) {
        return SZ_ERR_NULL;
    }

    *out_n = 0;

    if (n == 0) {
        return SZ_OK;
    }

    if (chunk_ids == NULL || block_ids == NULL || out_chunk == NULL ||
        out_first == NULL || out_last == NULL || out_item_start == NULL) {
        return SZ_ERR_NULL;
    }

    out_chunk[0] = chunk_ids[0];
    out_first[0] = block_ids[0];
    out_last[0] = block_ids[0];
    out_item_start[0] = 0;

    for (size_t i = 1; i < n; i++) {
        if (chunk_ids[i] < chunk_ids[i - 1] ||
            (chunk_ids[i] == chunk_ids[i - 1] && block_ids[i] <= block_ids[i - 1])) {
            return SZ_ERR_ARG;
        }

        if (chunk_ids[i] == out_chunk[m] &&
            (uint64_t)block_ids[i] <= out_last[m] + (uint64_t)gap_blocks + 1u) {
            out_last[m] = block_ids[i];
            continue;
        }

        m++;
        out_chunk[m] = chunk_ids[i];
        out_first[m] = block_ids[i];
        out_last[m] = block_ids[i];
        out_item_start[m] = i;
    }

    *out_n = m + 1;
    return SZ_OK;
}

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
) {
    size_t nval;
    size_t need_bytes;
    size_t count = 0;
    float vals[64];

    if (out_count == NULL) {
        return SZ_ERR_NULL;
    }

    *out_count = 0;

    if (n == 0) {
        return SZ_OK;
    }

    if (buf == NULL || offsets == NULL) {
        return SZ_ERR_NULL;
    }

    if (!block_layout(block_dim, rate, &nval, &need_bytes) || nval > 64) {
        return SZ_ERR_ARG;
    }

    if (block_nbytes != need_bytes) {
        return SZ_ERR_SIZE;
    }

    if (block_dim != 4) {
        return SZ_ERR_ARG;
    }

    for (size_t i = 0; i < n; i++) {
        if (offsets[i] > buf_size || buf_size - offsets[i] < block_nbytes) {
            return SZ_ERR_SIZE;
        }
    }

    if (szfp_fast_ok(block_dim, rate, block_nbytes)) {
        for (size_t i = 0; i < n; i++) {
            szfp_fast_decode(buf + offsets[i], block_nbytes, vals);

            count += count_in_range(vals, nval, lo, hi);
        }
    } else {
        bitstream* stream = stream_open((void*)buf, buf_size);
        zfp_stream* zfp;

        if (stream == NULL) {
            return SZ_ERR_ZFP;
        }

        zfp = zfp_stream_open(stream);
        if (zfp == NULL) {
            stream_close(stream);
            return SZ_ERR_ZFP;
        }

        zfp_stream_set_rate(zfp, rate, zfp_type_float, 3, 0);

        for (size_t i = 0; i < n; i++) {
            stream_rseek(stream, (bitstream_offset)offsets[i] * CHAR_BIT);
            zfp_decode_block_float_3(zfp, vals);

            count += count_in_range(vals, nval, lo, hi);
        }

        zfp_stream_close(zfp);
        stream_close(stream);
    }

    *out_count = count;
    return SZ_OK;
}

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
) {
    return szfp_count_offsets_range(
        buf,
        buf_size,
        offsets,
        n,
        block_nbytes,
        rate,
        block_dim,
        threshold,
        INFINITY,
        out_count
    );
}

/* add one decoded block to an aggregate for lo < x <= hi. Each original value
 * lies within eps of its decoded value r; in an IN block every original value
 * is known to match, while in a MAYBE block it is classified from r and eps. */
static void aggregate_block(
    const float* v,
    size_t n,
    double eps,
    int in_block,
    double lo,
    double hi,
    SzAggregate* acc
) {
    for (size_t i = 0; i < n; i++) {
        double r = (double)v[i];
        double a = r - eps;
        double b = r + eps;

        acc->abs_total += fabs(r) + eps;

        if (in_block || in_range(v[i], lo, hi)) {
            acc->rec_count++;
            acc->rec_sum += r;
        }

        if (in_block || (a > lo && b <= hi)) {
            /* the original value matches and lies in [a, b] within the range */
            acc->sure_count++;
            acc->sure_sum_lo += a > lo ? a : lo;
            acc->sure_sum_hi += b < hi ? b : hi;
        } else if (!(b <= lo || a > hi)) {
            /* it may or may not match; if it does, it lies in (lo, hi] and [a, b] */
            double low = a > lo ? a : lo;
            double high = b < hi ? b : hi;

            acc->maybe_count++;
            acc->maybe_sum_neg += low < 0.0 ? low : 0.0;
            acc->maybe_sum_pos += high > 0.0 ? high : 0.0;
            if (low < acc->maybe_min) {
                acc->maybe_min = low;
            }
            if (high > acc->maybe_max) {
                acc->maybe_max = high;
            }
        }
    }
}

int szfp_aggregate_offsets_range(
    const unsigned char* buf,
    size_t buf_size,
    const uint64_t* offsets,
    const float* eps,
    const unsigned char* in_block,
    size_t n,
    size_t block_nbytes,
    double rate,
    int block_dim,
    double lo,
    double hi,
    SzAggregate* out
) {
    size_t nval;
    size_t need_bytes;
    float vals[64];

    if (out == NULL) {
        return SZ_ERR_NULL;
    }

    memset(out, 0, sizeof(*out));
    out->maybe_min = INFINITY;
    out->maybe_max = -INFINITY;

    if (n == 0) {
        return SZ_OK;
    }

    if (buf == NULL || offsets == NULL || eps == NULL || in_block == NULL) {
        return SZ_ERR_NULL;
    }

    if (!block_layout(block_dim, rate, &nval, &need_bytes) || nval > 64 ||
        block_dim != 4) {
        return SZ_ERR_ARG;
    }

    if (block_nbytes != need_bytes) {
        return SZ_ERR_SIZE;
    }

    for (size_t i = 0; i < n; i++) {
        if (offsets[i] > buf_size || buf_size - offsets[i] < block_nbytes) {
            return SZ_ERR_SIZE;
        }
        if (!(eps[i] >= 0.0f) || !isfinite(eps[i])) {
            return SZ_ERR_ARG;
        }
    }

    if (szfp_fast_ok(block_dim, rate, block_nbytes)) {
        for (size_t i = 0; i < n; i++) {
            szfp_fast_decode(buf + offsets[i], block_nbytes, vals);
            aggregate_block(vals, nval, (double)eps[i], in_block[i] != 0, lo, hi, out);
        }
    } else {
        bitstream* stream = stream_open((void*)buf, buf_size);
        zfp_stream* zfp;

        if (stream == NULL) {
            return SZ_ERR_ZFP;
        }

        zfp = zfp_stream_open(stream);
        if (zfp == NULL) {
            stream_close(stream);
            return SZ_ERR_ZFP;
        }

        zfp_stream_set_rate(zfp, rate, zfp_type_float, 3, 0);

        for (size_t i = 0; i < n; i++) {
            stream_rseek(stream, (bitstream_offset)offsets[i] * CHAR_BIT);
            zfp_decode_block_float_3(zfp, vals);
            aggregate_block(vals, nval, (double)eps[i], in_block[i] != 0, lo, hi, out);
        }

        zfp_stream_close(zfp);
        stream_close(stream);
    }

    return SZ_OK;
}
