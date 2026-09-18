
#include <zfp.h>

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../core.h"

enum {
    SIDE = 32,
    VALUES = SIDE * SIDE * SIDE,
    BLOCK_VALUES = 64,
    BLOCKS = VALUES / BLOCK_VALUES,
    FULL_BLOCK_BYTES = 64,
    PACKED_BYTES = VALUES + 12 + 2 * BLOCKS,
    LEVELS = 4
};

static float input[VALUES];
static float sorted[VALUES];
static float decoded[LEVELS][BLOCKS][BLOCK_VALUES];
static double block_eps[LEVELS][BLOCKS];
static double chunk_eps[LEVELS];
static float block_min[BLOCKS];
static float block_max[BLOCKS];
static unsigned char packed[PACKED_BYTES];

static const double rates[LEVELS] = {2.0, 4.0, 6.0, 8.0};

static int decode_prefix(
    const unsigned char* source,
    size_t prefix_bytes,
    double rate,
    float output[BLOCK_VALUES]
) {
    unsigned char padded[FULL_BLOCK_BYTES] = {0};
    bitstream* bs = NULL;
    zfp_stream* zfp = NULL;
    size_t consumed = 0;

    if (prefix_bytes > FULL_BLOCK_BYTES)
        return 0;

    memcpy(padded, source, prefix_bytes);

    bs = stream_open(padded, sizeof(padded));
    if (bs == NULL)
        return 0;

    zfp = zfp_stream_open(bs);
    if (zfp == NULL) {
        stream_close(bs);
        return 0;
    }

    if (fabs(zfp_stream_set_rate(zfp, rate, zfp_type_float, 3, 0) - rate) < 1e-9) {
        zfp_stream_rewind(zfp);
        consumed = zfp_decode_block_float_3(zfp, output);
    }

    zfp_stream_close(zfp);
    stream_close(bs);

    return consumed != 0 && consumed <= prefix_bytes * 8u;
}

static void make_sample(int sample) {
    for (size_t i = 0; i < VALUES; ++i) {
        float x = (float)(i % SIDE);
        float y = (float)((i / SIDE) % SIDE);
        float z = (float)(i / (SIDE * SIDE));

        if (sample == 0) {
            /* Temperature-like smooth field. */
            input[i] = 280.0f + 0.13f * x + 0.09f * y + 0.06f * z +
                       0.15f * sinf(0.3f * x + 0.2f * y);
        } else if (sample == 1) {
            /* Signed wind-like field. */
            input[i] =
                6.0f * sinf(0.23f * x + 0.11f * z) - 4.0f * cosf(0.19f * y) + 0.07f * z;
        } else {
            /* Irregular field to avoid testing smooth data only. */
            input[i] = (float)((i * 73u + 19u) % 211u) * 0.19f - 17.0f;
        }
    }
}

static int compare_float(const void* a, const void* b) {
    float x = *(const float*)a;
    float y = *(const float*)b;
    return (x > y) - (x < y);
}

static int undecided_count(
    const float approx[BLOCK_VALUES],
    const float reference[BLOCK_VALUES],
    double eps,
    double threshold,
    size_t* unknown
) {
    size_t count = 0;

    for (size_t i = 0; i < BLOCK_VALUES; ++i) {
        double v = (double)approx[i];
        int truth = (double)reference[i] > threshold;

        if (v - eps > threshold) {
            if (!truth)
                return 0;
        } else if (v + eps <= threshold) {
            if (truth)
                return 0;
        } else {
            ++count;
        }
    }

    *unknown = count;
    return 1;
}

static int prepare_sample(int sample) {
    size_t written = 0;
    float cmin, cmax;
    double span, delta;
    SzResult result;

    make_sample(sample);

    result = szfp_pack_chunk(
        input, SIDE, SIDE, SIDE, 8.0, 4, packed, sizeof(packed), &written
    );

    if (result != SZ_OK || written != sizeof(packed)) {
        fprintf(
            stderr,
            "FAIL: packing sample %d, code=%d, size=%zu\n",
            sample,
            (int)result,
            written
        );
        return 0;
    }

    for (size_t b = 0; b < BLOCKS; ++b) {
        const unsigned char* payload = packed + b * FULL_BLOCK_BYTES;

        for (size_t level = 0; level < LEVELS; ++level) {
            size_t prefix_bytes = 16u * (level + 1u);

            if (!decode_prefix(
                    payload, prefix_bytes, rates[level], decoded[level][b]
                )) {
                fprintf(
                    stderr,
                    "FAIL: prefix decode sample=%d block=%zu "
                    "rate=%.0f\n",
                    sample,
                    b,
                    rates[level]
                );
                return 0;
            }
        }

        float bmin = INFINITY;
        float bmax = -INFINITY;

        for (size_t i = 0; i < BLOCK_VALUES; ++i) {
            float v = decoded[3][b][i];

            if (!isfinite(v)) {
                fprintf(stderr, "FAIL: non-finite decoded value\n");
                return 0;
            }

            if (v < bmin)
                bmin = v;
            if (v > bmax)
                bmax = v;
        }

        block_min[b] = bmin;
        block_max[b] = bmax;
    }

    const unsigned char* meta = packed + VALUES;
    const unsigned char* offsets = meta + 12u;

    memcpy(&cmin, meta, sizeof(float));
    memcpy(&cmax, meta + sizeof(float), sizeof(float));

    if (!isfinite(cmin) || !isfinite(cmax) || cmax < cmin) {
        fprintf(stderr, "FAIL: invalid metadata header\n");
        return 0;
    }

    span = (double)cmax - (double)cmin;
    delta = span / 255.0;

    for (size_t b = 0; b < BLOCKS; ++b) {
        double lower, upper;

        if (span == 0.0) {
            lower = cmin;
            upper = cmax;
        } else {
            lower = (double)cmin + ((double)offsets[2u * b] / 255.0) * span - delta;

            upper =
                (double)cmin + ((double)offsets[2u * b + 1u] / 255.0) * span + delta;
        }

        if ((double)block_min[b] < lower || (double)block_max[b] > upper) {
            fprintf(stderr, "FAIL: metadata does not cover block %zu\n", b);
            return 0;
        }
    }

    for (size_t level = 0; level < LEVELS; ++level) {
        double largest_chunk_error = 0.0;

        for (size_t b = 0; b < BLOCKS; ++b) {
            double largest_block_error = 0.0;

            for (size_t i = 0; i < BLOCK_VALUES; ++i) {
                double error =
                    fabs((double)decoded[3][b][i] - (double)decoded[level][b][i]);

                if (error > largest_block_error)
                    largest_block_error = error;
            }

            block_eps[level][b] = largest_block_error == 0.0
                                      ? 0.0
                                      : nextafter(largest_block_error, INFINITY);

            if (block_eps[level][b] > largest_chunk_error)
                largest_chunk_error = block_eps[level][b];
        }

        chunk_eps[level] =
            largest_chunk_error == 0.0 ? 0.0 : nextafter(largest_chunk_error, INFINITY);
    }

    for (size_t b = 0; b < BLOCKS; ++b) {
        memcpy(sorted + b * BLOCK_VALUES, decoded[3][b], sizeof(decoded[3][b]));
    }

    qsort(sorted, VALUES, sizeof(float), compare_float);
    return 1;
}

static int evaluate_threshold(double requested_selectivity) {
    const unsigned char* meta = packed + VALUES;
    const unsigned char* offsets = meta + 12u;

    float cmin, cmax;
    double span, delta, threshold;
    size_t desired, actual = 0;

    size_t maybe_blocks = 0;
    size_t static_bytes = 0;
    size_t oracle_chunk_bytes = 0;
    size_t oracle_block_bytes = 0;
    size_t full_chunk_blocks = 0;
    size_t full_block_blocks = 0;
    size_t unknown_chunk[3] = {0, 0, 0};

    memcpy(&cmin, meta, sizeof(float));
    memcpy(&cmax, meta + sizeof(float), sizeof(float));

    span = (double)cmax - (double)cmin;
    delta = span / 255.0;

    desired = (size_t)llround(requested_selectivity * (double)VALUES);

    if (desired < 1u)
        desired = 1u;
    if (desired >= VALUES)
        desired = VALUES - 1u;

    threshold = (double)sorted[VALUES - desired - 1u];

    for (size_t i = 0; i < VALUES; ++i) {
        if ((double)sorted[i] > threshold)
            ++actual;
    }

    for (size_t b = 0; b < BLOCKS; ++b) {
        double lower, upper;

        if (span == 0.0) {
            lower = cmin;
            upper = cmax;
        } else {
            lower = (double)cmin + ((double)offsets[2u * b] / 255.0) * span - delta;

            upper =
                (double)cmin + ((double)offsets[2u * b + 1u] / 255.0) * span + delta;
        }

        if (upper <= threshold) {
            if ((double)block_max[b] > threshold)
                return 0;
            continue;
        }

        if (lower > threshold) {
            if ((double)block_min[b] <= threshold)
                return 0;
            continue;
        }

        ++maybe_blocks;
        static_bytes += FULL_BLOCK_BYTES;

        size_t need_chunk = 0;
        size_t need_block = 0;

        for (size_t level = 0; level < LEVELS; ++level) {
            size_t pending_chunk = 0;
            size_t pending_block = 0;

            if (!undecided_count(
                    decoded[level][b],
                    decoded[3][b],
                    chunk_eps[level],
                    threshold,
                    &pending_chunk
                )) {
                fprintf(
                    stderr,
                    "FAIL: invalid chunk-bound decision "
                    "block=%zu level=%zu\n",
                    b,
                    level
                );
                return 0;
            }

            if (!undecided_count(
                    decoded[level][b],
                    decoded[3][b],
                    block_eps[level][b],
                    threshold,
                    &pending_block
                )) {
                fprintf(
                    stderr,
                    "FAIL: invalid block-bound decision "
                    "block=%zu level=%zu\n",
                    b,
                    level
                );
                return 0;
            }

            if (level < 3u)
                unknown_chunk[level] += pending_chunk;

            if (need_chunk == 0u && pending_chunk == 0u)
                need_chunk = 16u * (level + 1u);

            if (need_block == 0u && pending_block == 0u)
                need_block = 16u * (level + 1u);
        }

        if (need_chunk == 0u || need_block == 0u) {
            fprintf(
                stderr,
                "FAIL: 8-bit reference did not resolve "
                "block %zu\n",
                b
            );
            return 0;
        }

        oracle_chunk_bytes += need_chunk;
        oracle_block_bytes += need_block;

        if (need_chunk == FULL_BLOCK_BYTES)
            ++full_chunk_blocks;
        if (need_block == FULL_BLOCK_BYTES)
            ++full_block_blocks;
    }

    if (maybe_blocks == 0u) {
        printf(
            "target=%7.3f%% actual=%7.3f%% "
            "MAYBE=0 (metadata decides everything)\n",
            requested_selectivity * 100.0,
            100.0 * (double)actual / VALUES
        );
        return 1;
    }

    printf(
        "target=%7.3f%% actual=%7.3f%% MAYBE=%3zu | "
        "bytes static=%5zu chunk=%5zu block=%5zu | "
        "saving chunk=%5.2fx block=%5.2fx | "
        "need8 chunk=%3zu block=%3zu | "
        "U2/U4/U6=%zu/%zu/%zu\n",
        requested_selectivity * 100.0,
        100.0 * (double)actual / VALUES,
        maybe_blocks,
        static_bytes,
        oracle_chunk_bytes,
        oracle_block_bytes,
        (double)static_bytes / (double)oracle_chunk_bytes,
        (double)static_bytes / (double)oracle_block_bytes,
        full_chunk_blocks,
        full_block_blocks,
        unknown_chunk[0],
        unknown_chunk[1],
        unknown_chunk[2]
    );

    return 1;
}

int main(void) {
    const char* names[] = {"temperature-like", "wind-like", "irregular"};

    const double selectivities[] = {
        0.0001, /* 0.01% */
        0.001,  /* 0.1%  */
        0.01,   /* 1%    */
        0.10,   /* 10%   */
        0.50,   /* 50%   */
        0.90    /* 90%   */
    };

    for (int sample = 0; sample < 3; ++sample) {
        if (!prepare_sample(sample))
            return 1;

        printf("\n=== %s ===\n", names[sample]);

        for (size_t i = 0; i < sizeof(selectivities) / sizeof(selectivities[0]); ++i) {
            if (!evaluate_threshold(selectivities[i])) {
                fprintf(stderr, "FAIL: oracle evaluation\n");
                return 1;
            }
        }
    }

    puts("\nALL ORACLE B CHECKS PASSED");
    return 0;
}