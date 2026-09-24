#include <zfp.h>

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

enum {
    SZFP_OK = 0,
    SZFP_ERR_NULL = 1,
    SZFP_ERR_ARG = 2,
    SZFP_ERR_SIZE = 3,
    SZFP_ERR_MALLOC = 4,
    SZFP_ERR_ZFP = 5
};

static int mul_size(size_t a, size_t b, size_t* out) {
    if (out == NULL) {
        return 0;
    }

    if (a != 0 && b > SIZE_MAX / a) {
        return 0;
    }

    *out = a * b;
    return 1;
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
        return SZFP_ERR_NULL;
    }

    if (src_size == 0 || block_dim <= 0 || rate <= 0.0 || !isfinite(rate)) {
        return SZFP_ERR_ARG;
    }

    stream = stream_open((void*)src, src_size);
    if (stream == NULL) {
        return SZFP_ERR_ZFP;
    }

    zfp = zfp_stream_open(stream);
    if (zfp == NULL) {
        stream_close(stream);
        return SZFP_ERR_ZFP;
    }

    zfp_stream_set_rate(zfp, rate, zfp_type_float, 3, 0);
    zfp_stream_rewind(zfp);

    field = zfp_field_3d(
        out, zfp_type_float, (uint)block_dim, (uint)block_dim, (uint)block_dim
    );

    if (field == NULL) {
        zfp_stream_close(zfp);
        stream_close(stream);
        return SZFP_ERR_ZFP;
    }

    ret = zfp_decompress(zfp, field);

    zfp_field_free(field);
    zfp_stream_close(zfp);
    stream_close(stream);

    return ret == 0 ? SZFP_ERR_ZFP : SZFP_OK;
}

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
) {
    size_t total_blocks;
    size_t off_size;
    size_t need_size;
    unsigned char* states;
    int workers;

    if (meta == NULL || maybe_chunks == NULL || maybe_blocks == NULL ||
        out_maybe == NULL || out_in == NULL || out_out == NULL) {
        return SZFP_ERR_NULL;
    }

    *out_maybe = 0;
    *out_in = 0;
    *out_out = 0;

    if (n_chunk == 0 || blocks_per_chunk == 0 ||
        !mul_size(n_chunk, blocks_per_chunk, &total_blocks) ||
        maybe_cap < total_blocks || !mul_size(blocks_per_chunk, 2u, &off_size)) {
        return SZFP_ERR_ARG;
    }

    if (3u * sizeof(float) > SIZE_MAX - off_size) {
        return SZFP_ERR_SIZE;
    }

    need_size = 3u * sizeof(float) + off_size;

    if (meta_size != need_size) {
        return SZFP_ERR_ARG;
    }

    states = (unsigned char*)malloc(total_blocks);
    if (states == NULL) {
        return SZFP_ERR_MALLOC;
    }

    workers = threads > 0 ? threads : 1;

#ifdef _OPENMP
    if (threads <= 0) {
        workers = omp_get_max_threads();
    }
#pragma omp parallel for schedule(static) num_threads(workers)
#endif
    for (size_t cid = 0; cid < n_chunk; cid++) {
        const unsigned char* cm = meta + cid * meta_size;
        const unsigned char* offs = cm + 3u * sizeof(float);
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

        for (size_t bid = 0; bid < blocks_per_chunk; bid++) {
            double bmin;
            double bmax;
            unsigned char state = 2;

            if (span == 0.0) {
                bmin = cmin;
                bmax = cmax;
            } else {
                bmin = (double)cmin + ((double)offs[bid * 2u] / 255.0) * span - slack;

                bmax =
                    (double)cmin + ((double)offs[bid * 2u + 1u] / 255.0) * span + slack;
            }

            if (bmin - (double)eps > threshold) {
                state = 1;
            } else if (bmax + (double)eps <= threshold) {
                state = 0;
            }

            states[cid * blocks_per_chunk + bid] = state;
        }
    }

    {
        size_t n_in = 0;
        size_t n_out = 0;
        size_t n_maybe = 0;

        for (size_t i = 0; i < total_blocks; i++) {
            if (states[i] == 1) {
                n_in++;
            } else if (states[i] == 0) {
                n_out++;
            } else {
                maybe_chunks[n_maybe] = (uint32_t)(i / blocks_per_chunk);
                maybe_blocks[n_maybe] = (uint32_t)(i % blocks_per_chunk);
                n_maybe++;
            }
        }

        *out_maybe = n_maybe;
        *out_in = n_in;
        *out_out = n_out;
    }

    free(states);
    return SZFP_OK;
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
    size_t nval;
    size_t need_bytes;
    size_t scratch_vals;
    size_t scratch_bytes;
    size_t count;
    float* scratch;
    int failed;
    int workers;

    if (out_count == NULL) {
        return SZFP_ERR_NULL;
    }

    *out_count = 0;

    if (block_count == 0) {
        return SZFP_OK;
    }

    if (blocks == NULL) {
        return SZFP_ERR_NULL;
    }

    if (!block_layout(block_dim, rate, &nval, &need_bytes)) {
        return SZFP_ERR_ARG;
    }

    if (block_nbytes != need_bytes) {
        return SZFP_ERR_SIZE;
    }

    workers = threads > 0 ? threads : 1;

#ifdef _OPENMP
    if (threads <= 0) {
        workers = omp_get_max_threads();
    }
#endif

    if (workers <= 0 || !mul_size((size_t)workers, nval, &scratch_vals) ||
        !mul_size(scratch_vals, sizeof(float), &scratch_bytes)) {
        return SZFP_ERR_SIZE;
    }

    scratch = (float*)malloc(scratch_bytes);
    if (scratch == NULL) {
        return SZFP_ERR_MALLOC;
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

        if (code != SZFP_OK) {
            failed = 1;
            continue;
        }

        local_count = 0;

        for (size_t i = 0; i < nval; i++) {
            local_count += (double)vals[i] > threshold;
        }

        count += local_count;
    }

    free(scratch);

    if (failed) {
        return SZFP_ERR_ZFP;
    }

    *out_count = count;
    return SZFP_OK;
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
        return SZFP_OK;
    }

    if (blocks == NULL || out == NULL) {
        return SZFP_ERR_NULL;
    }

    if (!block_layout(block_dim, rate, &nval, &need_bytes)) {
        return SZFP_ERR_ARG;
    }

    if (block_nbytes != need_bytes) {
        return SZFP_ERR_SIZE;
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
            ) != SZFP_OK) {
            failed = 1;
        }
    }

    return failed ? SZFP_ERR_ZFP : SZFP_OK;
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
        return SZFP_ERR_NULL;
    }

    *out_n = 0;

    if (n == 0) {
        return SZFP_OK;
    }

    if (chunk_ids == NULL || block_ids == NULL || out_chunk == NULL ||
        out_first == NULL || out_last == NULL || out_item_start == NULL) {
        return SZFP_ERR_NULL;
    }

    out_chunk[0] = chunk_ids[0];
    out_first[0] = block_ids[0];
    out_last[0] = block_ids[0];
    out_item_start[0] = 0;

    for (size_t i = 1; i < n; i++) {
        if (chunk_ids[i] < chunk_ids[i - 1] ||
            (chunk_ids[i] == chunk_ids[i - 1] && block_ids[i] <= block_ids[i - 1])) {
            return SZFP_ERR_ARG;
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
    return SZFP_OK;
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
    size_t nval;
    size_t need_bytes;
    size_t count = 0;
    float vals[64];

    if (out_count == NULL) {
        return SZFP_ERR_NULL;
    }

    *out_count = 0;

    if (n == 0) {
        return SZFP_OK;
    }

    if (buf == NULL || offsets == NULL) {
        return SZFP_ERR_NULL;
    }

    if (!block_layout(block_dim, rate, &nval, &need_bytes) || nval > 64) {
        return SZFP_ERR_ARG;
    }

    if (block_nbytes != need_bytes) {
        return SZFP_ERR_SIZE;
    }

    if (block_dim != 4) {
        return SZFP_ERR_ARG;
    }

    for (size_t i = 0; i < n; i++) {
        if (offsets[i] > buf_size || buf_size - offsets[i] < block_nbytes) {
            return SZFP_ERR_SIZE;
        }
    }

    {
        bitstream* stream = stream_open((void*)buf, buf_size);
        zfp_stream* zfp;

        if (stream == NULL) {
            return SZFP_ERR_ZFP;
        }

        zfp = zfp_stream_open(stream);
        if (zfp == NULL) {
            stream_close(stream);
            return SZFP_ERR_ZFP;
        }

        zfp_stream_set_rate(zfp, rate, zfp_type_float, 3, 0);

        for (size_t i = 0; i < n; i++) {
            stream_rseek(stream, (bitstream_offset)offsets[i] * CHAR_BIT);
            zfp_decode_block_float_3(zfp, vals);

            for (size_t k = 0; k < nval; k++) {
                count += (double)vals[k] > threshold;
            }
        }

        zfp_stream_close(zfp);
        stream_close(stream);
    }

    *out_count = count;
    return SZFP_OK;
}