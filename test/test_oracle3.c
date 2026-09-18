
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
    LEVELS = 4,
    SEEDS = 5,
    SMOOTH_LEVELS = 5,
    SELECTIVITIES = 3,
    FAMILIES = 2
};

static const double rates[LEVELS] = {2.0, 4.0, 6.0, 8.0};

static const uint32_t seed_values[SEEDS] =
    {0x12345678u, 0x9e3779b9u, 0x243f6a88u, 0xb7e15162u, 0x87654321u};

static const int smoothing_passes[SMOOTH_LEVELS] = {0, 1, 2, 4, 8};

static const double selectivities[SELECTIVITIES] = {0.01, 0.10, 0.50};

typedef struct {
    size_t maybe;
    size_t need8_chunk;
    size_t need8_block;

    size_t static_bytes;
    size_t chunk_bytes;
    size_t block_bytes;
} OracleResult;

static float input[VALUES];
static float sorted[VALUES];
static float noise[VALUES];
static float noise_tmp[VALUES];

static float decoded[LEVELS][BLOCKS][BLOCK_VALUES];
static float block_min[BLOCKS];
static float block_max[BLOCKS];

static double block_eps[LEVELS][BLOCKS];
static double chunk_eps[LEVELS];

static unsigned char packed[PACKED_BYTES];

static OracleResult results[FAMILIES][SMOOTH_LEVELS][SELECTIVITIES][SEEDS];

static double neighbor_diff[FAMILIES][SMOOTH_LEVELS][SEEDS];

static double eps2[FAMILIES][SMOOTH_LEVELS][SEEDS];

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

    if (prefix_bytes == 0 || prefix_bytes > FULL_BLOCK_BYTES) {
        return 0;
    }

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

        consumed = zfp_decode_block_float_3(zfp, output);
    }

    zfp_stream_close(zfp);
    stream_close(bs);

    return consumed != 0 && consumed <= prefix_bytes * 8u;
}

static int make_noise(int family, uint32_t seed, int passes, double* neighbor_mad) {
    uint32_t state = seed;

    if (family == 1)
        state ^= 0xa5a5a5a5u;

    if (state == 0)
        state = 1;

    for (size_t i = 0; i < VALUES; ++i) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;

        noise[i] = 2.0f * (float)(state & 0x00ffffffu) / 16777215.0f - 1.0f;
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

                                sum += noise[j];
                            }
                        }
                    }

                    size_t i = x + SIDE * (y + SIDE * z);

                    noise_tmp[i] = (float)(sum / 27.0);
                }
            }
        }

        memcpy(noise, noise_tmp, sizeof(noise));
    }

    double mean = 0.0;
    double variance = 0.0;

    for (size_t i = 0; i < VALUES; ++i)
        mean += (double)noise[i];

    mean /= (double)VALUES;

    for (size_t i = 0; i < VALUES; ++i) {
        double d = (double)noise[i] - mean;
        variance += d * d;
    }

    double stddev = sqrt(variance / (double)VALUES);

    if (!(stddev > 0.0) || !isfinite(stddev))
        return 0;

    for (size_t i = 0; i < VALUES; ++i) {
        noise[i] = (float)(((double)noise[i] - mean) / stddev);
    }

    double total = 0.0;
    size_t pairs = 0;

    for (size_t z = 0; z < SIDE; ++z) {
        for (size_t y = 0; y < SIDE; ++y) {
            for (size_t x = 0; x < SIDE; ++x) {
                size_t i = x + SIDE * (y + SIDE * z);

                if (x + 1 < SIDE) {
                    total += fabs((double)noise[i] - (double)noise[i + 1]);
                    ++pairs;
                }

                if (y + 1 < SIDE) {
                    total += fabs((double)noise[i] - (double)noise[i + SIDE]);
                    ++pairs;
                }

                if (z + 1 < SIDE) {
                    total += fabs((double)noise[i] - (double)noise[i + SIDE * SIDE]);
                    ++pairs;
                }
            }
        }
    }

    *neighbor_mad = total / (double)pairs;
    return 1;
}

static int make_sample(int family, uint32_t seed, int passes, double* neighbor_mad) {
    if (!make_noise(family, seed, passes, neighbor_mad)) {
        return 0;
    }

    for (size_t i = 0; i < VALUES; ++i) {
        float x = (float)(i % SIDE);
        float y = (float)((i / SIDE) % SIDE);
        float z = (float)(i / (SIDE * SIDE));

        if (family == 0) {
            float base = 280.0f + 0.13f * x + 0.09f * y + 0.06f * z +
                         0.15f * sinf(0.3f * x + 0.2f * y);

            input[i] = base + 0.8f * noise[i];

        } else {
            float base =
                6.0f * sinf(0.23f * x + 0.11f * z) - 4.0f * cosf(0.19f * y) + 0.07f * z;

            input[i] = base + 1.5f * noise[i];
        }
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

static int prepare_sample(int family, uint32_t seed, int passes, double* neighbor_mad) {
    if (!make_sample(family, seed, passes, neighbor_mad)) {
        fprintf(stderr, "FAIL: noise generation\n");
        return 0;
    }

    size_t written = 0;

    SzResult result = szfp_pack_chunk(
        input, SIDE, SIDE, SIDE, 8.0, 4, packed, sizeof(packed), &written
    );

    if (result != SZ_OK || written != sizeof(packed)) {
        fprintf(stderr, "FAIL: packing, code=%d size=%zu\n", (int)result, written);
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
                    "FAIL: prefix decode "
                    "block=%zu rate=%.0f\n",
                    b,
                    rates[level]
                );
                return 0;
            }

            for (size_t i = 0; i < BLOCK_VALUES; ++i) {

                if (!isfinite(decoded[level][b][i])) {
                    fprintf(stderr, "FAIL: non-finite reconstruction\n");
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

        fprintf(stderr, "FAIL: invalid metadata\n");
        return 0;
    }

    double cmin = (double)cmin_f;
    double cmax = (double)cmax_f;

    for (size_t b = 0; b < BLOCKS; ++b) {
        double lower;
        double upper;

        metadata_interval(offsets, b, cmin, cmax, &lower, &upper);

        if ((double)block_min[b] < lower || (double)block_max[b] > upper) {

            fprintf(stderr, "FAIL: metadata misses block %zu\n", b);
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

            if (block_eps[level][b] > largest_chunk_error) {

                largest_chunk_error = block_eps[level][b];
            }
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
    size_t pending = 0;

    for (size_t i = 0; i < BLOCK_VALUES; ++i) {

        double v = (double)approx[i];

        int truth = (double)reference[i] > threshold;

        double lower = eps == 0.0 ? v : nextafter(v - eps, -INFINITY);

        double upper = eps == 0.0 ? v : nextafter(v + eps, INFINITY);

        if (lower > threshold) {
            if (!truth)
                return 0;

        } else if (upper <= threshold) {
            if (truth)
                return 0;

        } else {
            ++pending;
        }
    }

    *unknown = pending;
    return 1;
}

static int evaluate_threshold(double target_selectivity, OracleResult* out) {
    const unsigned char* meta = packed + VALUES;

    const unsigned char* offsets = meta + 12u;

    float cmin_f;
    float cmax_f;

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

    memset(out, 0, sizeof(*out));

    for (size_t b = 0; b < BLOCKS; ++b) {
        double lower;
        double upper;

        metadata_interval(offsets, b, cmin, cmax, &lower, &upper);

        /* Metadata-only OUT. */
        if (upper <= threshold) {
            if ((double)block_max[b] > threshold) {
                fprintf(stderr, "FAIL: false metadata OUT\n");
                return 0;
            }

            continue;
        }

        /* Metadata-only IN. */
        if (lower > threshold) {
            if ((double)block_min[b] <= threshold) {
                fprintf(stderr, "FAIL: false metadata IN\n");
                return 0;
            }

            continue;
        }

        ++out->maybe;

        out->static_bytes += FULL_BLOCK_BYTES;

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
                    "FAIL: unsafe chunk decision "
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
                    "FAIL: unsafe block decision "
                    "block=%zu level=%zu\n",
                    b,
                    level
                );
                return 0;
            }

            size_t prefix_bytes = 16u * (level + 1u);

            if (need_chunk == 0u && pending_chunk == 0u) {

                need_chunk = prefix_bytes;
            }

            if (need_block == 0u && pending_block == 0u) {

                need_block = prefix_bytes;
            }
        }

        if (need_chunk == 0u || need_block == 0u) {

            fprintf(stderr, "FAIL: unresolved at stored precision\n");
            return 0;
        }

        out->chunk_bytes += need_chunk;
        out->block_bytes += need_block;

        if (need_chunk == FULL_BLOCK_BYTES)
            ++out->need8_chunk;

        if (need_block == FULL_BLOCK_BYTES)
            ++out->need8_block;
    }

    return 1;
}

static void min_median_max(
    const double values[SEEDS],
    double* minimum,
    double* median,
    double* maximum
) {
    double tmp[SEEDS];

    memcpy(tmp, values, sizeof(tmp));

    for (int i = 0; i < SEEDS; ++i) {
        for (int j = i + 1; j < SEEDS; ++j) {
            if (tmp[j] < tmp[i]) {
                double swap = tmp[i];
                tmp[i] = tmp[j];
                tmp[j] = swap;
            }
        }
    }

    *minimum = tmp[0];
    *median = tmp[SEEDS / 2];
    *maximum = tmp[SEEDS - 1];
}

static void print_summary(int family, int smoothing_index, int selectivity_index) {
    double chunk_saving[SEEDS];
    double block_saving[SEEDS];
    double need8_fraction[SEEDS];
    double maybe_counts[SEEDS];

    size_t total_static = 0;
    size_t total_chunk = 0;

    for (int s = 0; s < SEEDS; ++s) {
        const OracleResult* r = &results[family][smoothing_index][selectivity_index][s];

        chunk_saving[s] = r->chunk_bytes == 0u
                              ? 1.0
                              : (double)r->static_bytes / (double)r->chunk_bytes;

        block_saving[s] = r->block_bytes == 0u
                              ? 1.0
                              : (double)r->static_bytes / (double)r->block_bytes;

        need8_fraction[s] =
            r->maybe == 0u ? 0.0 : 100.0 * (double)r->need8_chunk / (double)r->maybe;

        maybe_counts[s] = (double)r->maybe;

        total_static += r->static_bytes;
        total_chunk += r->chunk_bytes;
    }

    double cmin, cmed, cmax;
    double bmin, bmed, bmax;
    double fmin, fmed, fmax;
    double mmin, mmed, mmax;

    min_median_max(chunk_saving, &cmin, &cmed, &cmax);

    min_median_max(block_saving, &bmin, &bmed, &bmax);

    min_median_max(need8_fraction, &fmin, &fmed, &fmax);

    min_median_max(maybe_counts, &mmin, &mmed, &mmax);

    double pooled_saving =
        total_chunk == 0u ? 1.0 : (double)total_static / (double)total_chunk;

    printf(
        "sel=%5.0f%% | "
        "chunk x%.2f [%.2f, %.2f] | "
        "block x%.2f [%.2f, %.2f] | "
        "need8 med=%5.1f%% [%.1f, %.1f] | "
        "MAYBE med=%3.0f | "
        "pooled=%.2fx\n",

        100.0 * selectivities[selectivity_index],

        cmed,
        cmin,
        cmax,
        bmed,
        bmin,
        bmax,
        fmed,
        fmin,
        fmax,
        mmed,
        pooled_saving
    );
}

int main(void) {
    const char* family_names[FAMILIES] = {"temperature-like", "wind-like"};

    puts("Oracle B: five-seed smoothness sweep");
    puts("Exact = equal to fully decoded stored 8-bit data.");
    puts("Saving is MAYBE payload only, NOT query speedup.");
    puts("[min, max] describes variation across five seeds.");
    puts("No GCP access or storage-layout modification.");
    puts("");

    for (int family = 0; family < FAMILIES; ++family) {

        for (int seed_index = 0; seed_index < SEEDS; ++seed_index) {

            for (int p = 0; p < SMOOTH_LEVELS; ++p) {

                double mad = 0.0;

                if (!prepare_sample(
                        family, seed_values[seed_index], smoothing_passes[p], &mad
                    )) {

                    fprintf(
                        stderr,
                        "FAIL: family=%d seed=%d passes=%d\n",
                        family,
                        seed_index,
                        smoothing_passes[p]
                    );
                    return 1;
                }

                neighbor_diff[family][p][seed_index] = mad;

                eps2[family][p][seed_index] = chunk_eps[0];

                for (int t = 0; t < SELECTIVITIES; ++t) {

                    if (!evaluate_threshold(
                            selectivities[t], &results[family][p][t][seed_index]
                        )) {

                        fprintf(
                            stderr,
                            "FAIL: family=%d seed=%d "
                            "passes=%d selectivity=%.2f\n",
                            family,
                            seed_index,
                            smoothing_passes[p],
                            selectivities[t]
                        );
                        return 1;
                    }
                }
            }
        }
    }

    for (int family = 0; family < FAMILIES; ++family) {

        printf("\n========== %s ==========\n", family_names[family]);

        for (int p = 0; p < SMOOTH_LEVELS; ++p) {

            double mad_min, mad_med, mad_max;
            double eps_min, eps_med, eps_max;

            min_median_max(neighbor_diff[family][p], &mad_min, &mad_med, &mad_max);

            min_median_max(eps2[family][p], &eps_min, &eps_med, &eps_max);

            printf(
                "\npasses=%d | "
                "neighbor diff med=%.4f "
                "[%.4f, %.4f] | "
                "2-bit chunk eps med=%.6g "
                "[%.6g, %.6g]\n",

                smoothing_passes[p],

                mad_med,
                mad_min,
                mad_max,

                eps_med,
                eps_min,
                eps_max
            );

            for (int t = 0; t < SELECTIVITIES; ++t) {

                print_summary(family, p, t);
            }
        }
    }

    puts("\nALL MULTI-SEED ORACLE CHECKS PASSED");
    return 0;
}