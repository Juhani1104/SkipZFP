
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
    LEVELS = 4,
    PACKED_BYTES = VALUES + 12 + 2 * BLOCKS
};

static const double rates[LEVELS] = {2.0, 4.0, 6.0, 8.0};

static float input[VALUES];
static float sorted[VALUES];
static float noise_field[VALUES];
static float noise_tmp[VALUES];

static float decoded[LEVELS][BLOCKS][BLOCK_VALUES];
static float block_min[BLOCKS];
static float block_max[BLOCKS];

static double block_eps[LEVELS][BLOCKS];
static double chunk_eps[LEVELS];
static double noise_neighbor_mad;

static unsigned char packed[PACKED_BYTES];

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

    if (prefix_bytes == 0 || prefix_bytes > FULL_BLOCK_BYTES)
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

    double actual_rate = zfp_stream_set_rate(zfp, rate, zfp_type_float, 3, 0);

    if (fabs(actual_rate - rate) < 1e-9) {
        zfp_stream_rewind(zfp);

        /* Low-level single-block decoder. */
        consumed = zfp_decode_block_float_3(zfp, output);
    }

    zfp_stream_close(zfp);
    stream_close(bs);

    return consumed != 0 && consumed <= prefix_bytes * 8u;
}

static int make_spatial_noise(int sample, int passes) {
    uint32_t state = sample == 0 ? 0x12345678u : 0x87654321u;

    /*
     * Same initial noise within each sample family,
     * regardless of the number of smoothing passes.
     */
    for (size_t i = 0; i < VALUES; ++i) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;

        noise_field[i] = 2.0f * (float)(state & 0x00ffffffu) / 16777215.0f - 1.0f;
    }

    for (int pass = 0; pass < passes; ++pass) {
        for (size_t z = 0; z < SIDE; ++z) {
            size_t zs[3] = {z == 0 ? SIDE - 1 : z - 1, z, z + 1 == SIDE ? 0 : z + 1};

            for (size_t y = 0; y < SIDE; ++y) {
                size_t ys[3] = {
                    y == 0 ? SIDE - 1 : y - 1, y, y + 1 == SIDE ? 0 : y + 1
                };

                for (size_t x = 0; x < SIDE; ++x) {
                    size_t xs[3] = {
                        x == 0 ? SIDE - 1 : x - 1, x, x + 1 == SIDE ? 0 : x + 1
                    };

                    double sum = 0.0;

                    for (int dz = 0; dz < 3; ++dz) {
                        for (int dy = 0; dy < 3; ++dy) {
                            for (int dx = 0; dx < 3; ++dx) {
                                size_t j = xs[dx] + SIDE * (ys[dy] + SIDE * zs[dz]);

                                sum += noise_field[j];
                            }
                        }
                    }

                    size_t i = x + SIDE * (y + SIDE * z);
                    noise_tmp[i] = (float)(sum / 27.0);
                }
            }
        }

        memcpy(noise_field, noise_tmp, sizeof(noise_field));
    }

    double mean = 0.0;
    double variance = 0.0;

    for (size_t i = 0; i < VALUES; ++i)
        mean += (double)noise_field[i];

    mean /= (double)VALUES;

    for (size_t i = 0; i < VALUES; ++i) {
        double d = (double)noise_field[i] - mean;
        variance += d * d;
    }

    double stddev = sqrt(variance / (double)VALUES);

    if (!(stddev > 0.0) || !isfinite(stddev))
        return 0;

    for (size_t i = 0; i < VALUES; ++i) {
        noise_field[i] = (float)(((double)noise_field[i] - mean) / stddev);
    }

    double total_diff = 0.0;
    size_t pairs = 0;

    for (size_t z = 0; z < SIDE; ++z) {
        for (size_t y = 0; y < SIDE; ++y) {
            for (size_t x = 0; x < SIDE; ++x) {
                size_t i = x + SIDE * (y + SIDE * z);

                if (x + 1 < SIDE) {
                    total_diff +=
                        fabs((double)noise_field[i] - (double)noise_field[i + 1]);
                    ++pairs;
                }

                if (y + 1 < SIDE) {
                    total_diff +=
                        fabs((double)noise_field[i] - (double)noise_field[i + SIDE]);
                    ++pairs;
                }

                if (z + 1 < SIDE) {
                    total_diff += fabs(
                        (double)noise_field[i] - (double)noise_field[i + SIDE * SIDE]
                    );
                    ++pairs;
                }
            }
        }
    }

    noise_neighbor_mad = total_diff / (double)pairs;
    return 1;
}

static int make_sample(int sample, int smooth_passes) {
    for (size_t i = 0; i < VALUES; ++i) {
        float x = (float)(i % SIDE);
        float y = (float)((i / SIDE) % SIDE);
        float z = (float)(i / (SIDE * SIDE));

        if (sample == 0) {
            input[i] = 280.0f + 0.13f * x + 0.09f * y + 0.06f * z +
                       0.15f * sinf(0.3f * x + 0.2f * y);

        } else if (sample == 1) {
            input[i] =
                6.0f * sinf(0.23f * x + 0.11f * z) - 4.0f * cosf(0.19f * y) + 0.07f * z;

        } else {
            /* Unchanged irregular control. */
            input[i] = (float)((i * 73u + 19u) % 211u) * 0.19f - 17.0f;
        }
    }

    noise_neighbor_mad = 0.0;

    if (sample != 2 && smooth_passes >= 0) {
        if (!make_spatial_noise(sample, smooth_passes))
            return 0;

        float amplitude = sample == 0 ? 0.8f : 1.5f;

        for (size_t i = 0; i < VALUES; ++i)
            input[i] += amplitude * noise_field[i];
    }

    return 1;
}

static int compare_float(const void* a, const void* b) {
    float x = *(const float*)a;
    float y = *(const float*)b;

    return (x > y) - (x < y);
}

static void metadata_interval(
    const unsigned char* offsets,
    size_t block_id,
    double cmin,
    double cmax,
    double* lower,
    double* upper
) {
    double span = cmax - cmin;

    if (span == 0.0) {
        *lower = cmin;
        *upper = cmax;
        return;
    }

    double delta = span / 255.0;

    double lo = cmin + ((double)offsets[2u * block_id] / 255.0) * span - delta;

    double hi = cmin + ((double)offsets[2u * block_id + 1u] / 255.0) * span + delta;

    *lower = nextafter(lo, -INFINITY);
    *upper = nextafter(hi, INFINITY);
}

static int prepare_sample(int sample, int smooth_passes) {
    size_t written = 0;

    if (!make_sample(sample, smooth_passes)) {
        fprintf(stderr, "FAIL: noise generation\n");
        return 0;
    }

    SzResult result = szfp_pack_chunk(
        input, SIDE, SIDE, SIDE, 8.0, 4, packed, sizeof(packed), &written
    );

    if (result != SZ_OK || written != sizeof(packed)) {
        fprintf(
            stderr,
            "FAIL: packing sample=%d passes=%d code=%d size=%zu\n",
            sample,
            smooth_passes,
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
                    stderr, "FAIL: prefix decode block=%zu rate=%.0f\n", b, rates[level]
                );
                return 0;
            }

            for (size_t i = 0; i < BLOCK_VALUES; ++i) {
                if (!isfinite(decoded[level][b][i])) {
                    fprintf(
                        stderr,
                        "FAIL: non-finite decoded value "
                        "block=%zu rate=%.0f\n",
                        b,
                        rates[level]
                    );
                    return 0;
                }
            }
        }

        float bmin = INFINITY;
        float bmax = -INFINITY;

        for (size_t i = 0; i < BLOCK_VALUES; ++i) {
            float v = decoded[3][b][i];

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

    float cmin_f;
    float cmax_f;

    memcpy(&cmin_f, meta, sizeof(float));
    memcpy(&cmax_f, meta + sizeof(float), sizeof(float));

    if (!isfinite(cmin_f) || !isfinite(cmax_f) || cmax_f < cmin_f) {
        fprintf(stderr, "FAIL: invalid metadata header\n");
        return 0;
    }

    double cmin = (double)cmin_f;
    double cmax = (double)cmax_f;

    for (size_t b = 0; b < BLOCKS; ++b) {
        double lower, upper;

        metadata_interval(offsets, b, cmin, cmax, &lower, &upper);

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
            /* Certified IN. */
            if (!truth)
                return 0;

        } else if (v + eps <= threshold) {
            /* Certified OUT. */
            if (truth)
                return 0;

        } else {
            ++count;
        }
    }

    *unknown = count;
    return 1;
}

static int evaluate_threshold(double target_selectivity) {
    const unsigned char* meta = packed + VALUES;
    const unsigned char* offsets = meta + 12u;

    float cmin_f, cmax_f;

    memcpy(&cmin_f, meta, sizeof(float));
    memcpy(&cmax_f, meta + sizeof(float), sizeof(float));

    double cmin = (double)cmin_f;
    double cmax = (double)cmax_f;

    size_t desired = (size_t)llround(target_selectivity * (double)VALUES);

    if (desired < 1u)
        desired = 1u;

    if (desired >= VALUES)
        desired = VALUES - 1u;

    double threshold = (double)sorted[VALUES - desired - 1u];

    size_t actual = 0;

    for (size_t i = 0; i < VALUES; ++i) {
        if ((double)sorted[i] > threshold)
            ++actual;
    }

    size_t maybe_blocks = 0;
    size_t static_bytes = 0;
    size_t oracle_chunk_bytes = 0;
    size_t oracle_block_bytes = 0;

    size_t histogram_chunk[LEVELS] = {0};
    size_t need8_block = 0;
    size_t unknown_chunk[3] = {0};

    for (size_t b = 0; b < BLOCKS; ++b) {
        double lower, upper;

        metadata_interval(offsets, b, cmin, cmax, &lower, &upper);

        if (upper <= threshold) {
            if ((double)block_max[b] > threshold) {
                fprintf(stderr, "FAIL: false metadata OUT\n");
                return 0;
            }
            continue;
        }

        if (lower > threshold) {
            if ((double)block_min[b] <= threshold) {
                fprintf(stderr, "FAIL: false metadata IN\n");
                return 0;
            }
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
                    "FAIL: unsafe chunk-bound decision "
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
                    "FAIL: unsafe block-bound decision "
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
            fprintf(stderr, "FAIL: highest precision did not resolve block %zu\n", b);
            return 0;
        }

        oracle_chunk_bytes += need_chunk;
        oracle_block_bytes += need_block;

        histogram_chunk[need_chunk / 16u - 1u]++;

        if (need_block == FULL_BLOCK_BYTES)
            ++need8_block;
    }

    if (maybe_blocks == 0u) {
        printf(
            "target=%6.2f%% actual=%6.2f%% MAYBE=0\n",
            target_selectivity * 100.0,
            100.0 * (double)actual / (double)VALUES
        );
        return 1;
    }

    printf(
        "target=%6.2f%% actual=%6.2f%% MAYBE=%3zu | "
        "bytes %5zu -> %5zu / %5zu | "
        "saving %4.2fx / %4.2fx | "
        "L2/4/6/8=%3zu/%3zu/%3zu/%3zu | "
        "block_need8=%3zu | U2/4/6=%zu/%zu/%zu\n",

        target_selectivity * 100.0,
        100.0 * (double)actual / (double)VALUES,
        maybe_blocks,

        static_bytes,
        oracle_chunk_bytes,
        oracle_block_bytes,

        (double)static_bytes / (double)oracle_chunk_bytes,
        (double)static_bytes / (double)oracle_block_bytes,

        histogram_chunk[0],
        histogram_chunk[1],
        histogram_chunk[2],
        histogram_chunk[3],

        need8_block,

        unknown_chunk[0],
        unknown_chunk[1],
        unknown_chunk[2]
    );

    return 1;
}

int main(void) {
    const char* names[] = {"temperature-like", "wind-like", "irregular"};

    const int smoothness[] = {-1, 0, 1, 2, 4, 8};

    const double selectivities[] = {
        0.0001, /* 0.01% */
        0.001,  /* 0.1%  */
        0.01,   /* 1%    */
        0.10,   /* 10%   */
        0.50,   /* 50%   */
        0.90    /* 90%   */
    };

    for (int sample = 0; sample < 3; ++sample) {
        size_t runs = sample == 2 ? 1u : sizeof(smoothness) / sizeof(smoothness[0]);

        for (size_t s = 0; s < runs; ++s) {
            int passes = sample == 2 ? -1 : smoothness[s];

            if (!prepare_sample(sample, passes))
                return 1;

            if (sample == 2) {
                printf("\n=== %s: unchanged control ===\n", names[sample]);

            } else if (passes == -1) {
                printf("\n=== %s: original, no added noise ===\n", names[sample]);

            } else {
                printf(
                    "\n=== %s: smoothing passes=%d, "
                    "noise neighbor diff=%.4f ===\n",
                    names[sample],
                    passes,
                    noise_neighbor_mad
                );
            }

            printf(
                "chunk observed eps: "
                "2-bit=%.6g 4-bit=%.6g 6-bit=%.6g\n",
                chunk_eps[0],
                chunk_eps[1],
                chunk_eps[2]
            );

            for (size_t i = 0; i < sizeof(selectivities) / sizeof(selectivities[0]);
                 ++i) {
                if (!evaluate_threshold(selectivities[i])) {
                    fprintf(stderr, "FAIL: sample=%d passes=%d\n", sample, passes);
                    return 1;
                }
            }
        }
    }

    puts("\nALL SMOOTHNESS ORACLE CHECKS PASSED");
    return 0;
}