
#include <zfp.h>

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core.h"



enum {
    SIDE = 32,
    VALUES = SIDE * SIDE * SIDE,
    BLOCK_VALUES = 64,
    BLOCKS = VALUES / BLOCK_VALUES,
    FULL_BYTES = 64,
    PACKED_BYTES = VALUES + 12 + 2 * BLOCKS,
    LEVELS = 4,
    SEEDS = 5,
    PASSES = 5,
    QUERIES = 3,
    FAMILIES = 2,
    TOLERANCES = 2
};

static const double rates[LEVELS] = {
    2.0, 4.0, 6.0, 8.0
};

static const uint32_t seed_values[SEEDS] = {
    0x12345678u,
    0x9e3779b9u,
    0x243f6a88u,
    0xb7e15162u,
    0x87654321u
};

static const int smooth_passes[PASSES] = {
    0, 1, 2, 4, 8
};

static const double selectivities[QUERIES] = {
    0.01, 0.10, 0.50
};


static const size_t tolerance_divisor[TOLERANCES] = {
    100u, 1000u
};

typedef struct {
    size_t in;
    size_t unknown;
} Decision;

typedef struct {
    size_t bytes;
    size_t lower;
    size_t unknown;
    size_t max_level;
} AnytimeResult;

typedef struct {
    size_t maybe;
    size_t truth;
    size_t static_bytes;
    size_t exact_bytes;

    AnytimeResult uniform[TOLERANCES];
    AnytimeResult adaptive[TOLERANCES];
} QueryResult;

static float input[VALUES];
static float noise[VALUES];
static float noise_tmp[VALUES];
static float sorted[VALUES];

static unsigned char packed[PACKED_BYTES];

static float decoded[LEVELS][BLOCKS][BLOCK_VALUES];
static double chunk_eps[LEVELS];

static Decision decisions[LEVELS][BLOCKS];
static size_t maybe_ids[BLOCKS];
static size_t maybe_count;
static size_t metadata_in;

static float block_min[BLOCKS];
static float block_max[BLOCKS];

static QueryResult results[FAMILIES][PASSES][QUERIES][SEEDS];




static int decode_prefix(
    const unsigned char *source,
    size_t prefix_bytes,
    double rate,
    float output[BLOCK_VALUES]
) {
    unsigned char padded[FULL_BYTES] = {0};
    bitstream *bs = NULL;
    zfp_stream *zfp = NULL;
    size_t consumed = 0;

    if (prefix_bytes == 0 || prefix_bytes > FULL_BYTES)
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

    double actual = zfp_stream_set_rate(
        zfp, rate, zfp_type_float, 3, 0
    );

    if (fabs(actual - rate) < 1e-9) {
        zfp_stream_rewind(zfp);
        consumed = zfp_decode_block_float_3(zfp, output);
    }

    zfp_stream_close(zfp);
    stream_close(bs);

    return consumed != 0 &&
           consumed <= prefix_bytes * 8u;
}




static int make_noise(
    int family,
    uint32_t seed,
    int passes
) {
    uint32_t state = seed;

    if (family == 1)
        state ^= 0xa5a5a5a5u;

    if (state == 0u)
        state = 1u;

    for (size_t i = 0; i < VALUES; ++i) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;

        noise[i] =
            2.0f * (float)(state & 0x00ffffffu) /
            16777215.0f - 1.0f;
    }

    for (int pass = 0; pass < passes; ++pass) {
        for (size_t z = 0; z < SIDE; ++z) {
            size_t zs[3] = {
                z == 0 ? SIDE - 1 : z - 1,
                z,
                z + 1 == SIDE ? 0 : z + 1
            };

            for (size_t y = 0; y < SIDE; ++y) {
                size_t ys[3] = {
                    y == 0 ? SIDE - 1 : y - 1,
                    y,
                    y + 1 == SIDE ? 0 : y + 1
                };

                for (size_t x = 0; x < SIDE; ++x) {
                    size_t xs[3] = {
                        x == 0 ? SIDE - 1 : x - 1,
                        x,
                        x + 1 == SIDE ? 0 : x + 1
                    };

                    double sum = 0.0;

                    for (int dz = 0; dz < 3; ++dz) {
                        for (int dy = 0; dy < 3; ++dy) {
                            for (int dx = 0; dx < 3; ++dx) {
                                size_t j =
                                    xs[dx] +
                                    SIDE * (ys[dy] + SIDE * zs[dz]);

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
        noise[i] = (float)(
            ((double)noise[i] - mean) / stddev
        );
    }

    return 1;
}


static int make_sample(
    int family,
    uint32_t seed,
    int passes
) {
    if (!make_noise(family, seed, passes))
        return 0;

    for (size_t i = 0; i < VALUES; ++i) {
        float x = (float)(i % SIDE);
        float y = (float)((i / SIDE) % SIDE);
        float z = (float)(i / (SIDE * SIDE));

        if (family == 0) {
            float base =
                280.0f +
                0.13f * x +
                0.09f * y +
                0.06f * z +
                0.15f * sinf(0.3f * x + 0.2f * y);

            input[i] = base + 0.8f * noise[i];

        } else {
            float base =
                6.0f * sinf(0.23f * x + 0.11f * z) -
                4.0f * cosf(0.19f * y) +
                0.07f * z;

            input[i] = base + 1.5f * noise[i];
        }
    }

    return 1;
}


static int compare_float(const void *a, const void *b) {
    float x = *(const float *)a;
    float y = *(const float *)b;

    return (x > y) - (x < y);
}




static int prepare_sample(
    int family,
    uint32_t seed,
    int passes
) {
    if (!make_sample(family, seed, passes))
        return 0;

    size_t written = 0;

    SzResult result = szfp_pack_chunk(
        input,
        SIDE, SIDE, SIDE,
        8.0, 4,
        packed,
        sizeof(packed),
        &written
    );

    if (result != SZ_OK || written != sizeof(packed)) {
        fprintf(
            stderr,
            "FAIL: packing code=%d size=%zu\n",
            (int)result, written
        );
        return 0;
    }

    for (size_t b = 0; b < BLOCKS; ++b) {
        const unsigned char *payload =
            packed + b * FULL_BYTES;

        for (size_t level = 0; level < LEVELS; ++level) {
            size_t bytes = 16u * (level + 1u);

            if (!decode_prefix(
                    payload,
                    bytes,
                    rates[level],
                    decoded[level][b]
                )) {
                fprintf(
                    stderr,
                    "FAIL: prefix decode block=%zu level=%zu\n",
                    b, level
                );
                return 0;
            }

            for (size_t i = 0; i < BLOCK_VALUES; ++i) {
                if (!isfinite(decoded[level][b][i])) {
                    fprintf(
                        stderr,
                        "FAIL: non-finite decoded value\n"
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

    for (size_t level = 0; level < LEVELS; ++level) {
        double maximum = 0.0;

        for (size_t b = 0; b < BLOCKS; ++b) {
            for (size_t i = 0; i < BLOCK_VALUES; ++i) {
                double error = fabs(
                    (double)decoded[3][b][i] -
                    (double)decoded[level][b][i]
                );

                if (error > maximum)
                    maximum = error;
            }
        }

        chunk_eps[level] =
            maximum == 0.0
            ? 0.0
            : nextafter(maximum, INFINITY);
    }

    /* At full stored precision, the reference error is zero. */
    if (chunk_eps[3] != 0.0)
        return 0;

    for (size_t b = 0; b < BLOCKS; ++b) {
        memcpy(
            sorted + b * BLOCK_VALUES,
            decoded[3][b],
            sizeof(decoded[3][b])
        );
    }

    qsort(sorted, VALUES, sizeof(float), compare_float);

    return 1;
}




static void metadata_interval(
    const unsigned char *offsets,
    size_t b,
    double cmin,
    double cmax,
    double *lower,
    double *upper
) {
    double span = cmax - cmin;

    if (span == 0.0) {
        *lower = cmin;
        *upper = cmax;
        return;
    }

    double delta = span / 255.0;

    double lo =
        cmin +
        ((double)offsets[2u * b] / 255.0) * span -
        delta;

    double hi =
        cmin +
        ((double)offsets[2u * b + 1u] / 255.0) * span +
        delta;

    *lower = nextafter(lo, -INFINITY);
    *upper = nextafter(hi, INFINITY);
}




static int build_decisions(
    size_t block_id,
    double threshold
) {
    for (size_t level = 0; level < LEVELS; ++level) {
        decisions[level][block_id].in = 0;
        decisions[level][block_id].unknown = 0;
    }

    for (size_t i = 0; i < BLOCK_VALUES; ++i) {
        double lower = -INFINITY;
        double upper = INFINITY;

        double truth_value =
            (double)decoded[3][block_id][i];

        for (size_t level = 0; level < LEVELS; ++level) {
            double v =
                (double)decoded[level][block_id][i];

            double eps = chunk_eps[level];

            double lo = eps == 0.0
                ? v
                : nextafter(v - eps, -INFINITY);

            double hi = eps == 0.0
                ? v
                : nextafter(v + eps, INFINITY);

            if (lo > lower)
                lower = lo;

            if (hi < upper)
                upper = hi;

            if (lower > upper ||
                truth_value < lower ||
                truth_value > upper) {
                fprintf(
                    stderr,
                    "FAIL: invalid certified interval "
                    "block=%zu level=%zu\n",
                    block_id, level
                );
                return 0;
            }

            if (lower > threshold) {
                if (!(truth_value > threshold))
                    return 0;

                ++decisions[level][block_id].in;

            } else if (upper <= threshold) {
                if (truth_value > threshold)
                    return 0;

            } else {
                ++decisions[level][block_id].unknown;
            }
        }
    }

    for (size_t level = 1; level < LEVELS; ++level) {
        if (decisions[level][block_id].in <
                decisions[level - 1u][block_id].in ||
            decisions[level][block_id].unknown >
                decisions[level - 1u][block_id].unknown) {

            fprintf(stderr, "FAIL: nonmonotone decisions\n");
            return 0;
        }
    }

    if (decisions[3][block_id].unknown != 0u) {
        fprintf(stderr, "FAIL: unresolved at 8 bits\n");
        return 0;
    }

    return 1;
}




static int meets_tolerance(
    size_t lower,
    size_t unknown,
    size_t divisor
) {
    return unknown <= lower / divisor;
}


static int valid_answer(
    size_t lower,
    size_t unknown,
    size_t truth
) {
    return lower <= truth &&
           unknown <= VALUES - lower &&
           truth - lower <= unknown;
}




static int evaluate_uniform(
    size_t divisor,
    size_t truth,
    AnytimeResult *out
) {
    for (size_t level = 0; level < LEVELS; ++level) {
        size_t lower = metadata_in;
        size_t unknown = 0;

        for (size_t j = 0; j < maybe_count; ++j) {
            size_t b = maybe_ids[j];

            lower += decisions[level][b].in;
            unknown += decisions[level][b].unknown;
        }

        if (!valid_answer(lower, unknown, truth)) {
            fprintf(stderr, "FAIL: uniform COUNT interval\n");
            return 0;
        }

        if (meets_tolerance(lower, unknown, divisor)) {
            out->bytes =
                maybe_count * 16u * (level + 1u);

            out->lower = lower;
            out->unknown = unknown;
            out->max_level = level + 1u;

            return 1;
        }
    }

    fprintf(stderr, "FAIL: uniform did not converge\n");
    return 0;
}




static int evaluate_adaptive(
    size_t divisor,
    size_t truth,
    AnytimeResult *out
) {
    size_t current_level[BLOCKS] = {0};

    size_t lower = metadata_in;
    size_t unknown = 0;
    size_t bytes = maybe_count * 16u;
    size_t max_level = maybe_count == 0u ? 0u : 1u;

    for (size_t j = 0; j < maybe_count; ++j) {
        size_t b = maybe_ids[j];

        lower += decisions[0][b].in;
        unknown += decisions[0][b].unknown;
    }

    if (!valid_answer(lower, unknown, truth))
        return 0;

    while (!meets_tolerance(lower, unknown, divisor)) {
        size_t best_j = maybe_count;
        size_t best_target = 0;
        size_t best_gain = 0;
        size_t best_steps = 1;

        for (size_t j = 0; j < maybe_count; ++j) {
            size_t b = maybe_ids[j];
            size_t current = current_level[j];

            for (size_t target = current + 1u;
                 target < LEVELS;
                 ++target) {

                size_t before =
                    decisions[current][b].unknown;

                size_t after =
                    decisions[target][b].unknown;

                if (after > before) {
                    fprintf(stderr, "FAIL: unknown increased\n");
                    return 0;
                }

                size_t gain = before - after;
                size_t steps = target - current;

                if (gain == 0u)
                    continue;


                if (best_j == maybe_count ||
                    gain * best_steps >
                        best_gain * steps ||
                    (gain * best_steps ==
                        best_gain * steps &&
                     steps < best_steps)) {

                    best_j = j;
                    best_target = target;
                    best_gain = gain;
                    best_steps = steps;
                }
            }
        }

        if (best_j == maybe_count) {
            fprintf(stderr, "FAIL: no refinement available\n");
            return 0;
        }

        size_t b = maybe_ids[best_j];
        size_t old = current_level[best_j];

        size_t gained_in =
            decisions[best_target][b].in -
            decisions[old][b].in;

        size_t resolved =
            decisions[old][b].unknown -
            decisions[best_target][b].unknown;

        lower += gained_in;
        unknown -= resolved;
        bytes += 16u * (best_target - old);

        current_level[best_j] = best_target;

        if (best_target + 1u > max_level)
            max_level = best_target + 1u;

        if (!valid_answer(lower, unknown, truth)) {
            fprintf(stderr, "FAIL: adaptive COUNT interval\n");
            return 0;
        }
    }

    out->bytes = bytes;
    out->lower = lower;
    out->unknown = unknown;
    out->max_level = max_level;

    return 1;
}




static int evaluate_query(
    double target_selectivity,
    QueryResult *out
) {
    memset(out, 0, sizeof(*out));

    const unsigned char *meta = packed + VALUES;
    const unsigned char *offsets = meta + 12u;

    float cmin_f, cmax_f;

    memcpy(&cmin_f, meta, sizeof(float));
    memcpy(
        &cmax_f,
        meta + sizeof(float),
        sizeof(float)
    );

    if (!isfinite(cmin_f) ||
        !isfinite(cmax_f) ||
        cmax_f < cmin_f) {
        fprintf(stderr, "FAIL: invalid metadata\n");
        return 0;
    }

    double cmin = (double)cmin_f;
    double cmax = (double)cmax_f;

    size_t desired = (size_t)llround(
        target_selectivity * (double)VALUES
    );

    if (desired < 1u)
        desired = 1u;

    if (desired >= VALUES)
        desired = VALUES - 1u;

    double threshold =
        (double)sorted[VALUES - desired - 1u];

    size_t truth = 0;

    for (size_t i = 0; i < VALUES; ++i) {
        if ((double)sorted[i] > threshold)
            ++truth;
    }

    maybe_count = 0;
    metadata_in = 0;

    for (size_t b = 0; b < BLOCKS; ++b) {
        double lo, hi;

        metadata_interval(
            offsets,
            b,
            cmin,
            cmax,
            &lo,
            &hi
        );

        if ((double)block_min[b] < lo ||
            (double)block_max[b] > hi) {
            fprintf(stderr, "FAIL: metadata misses block\n");
            return 0;
        }

        if (hi <= threshold) {
            if ((double)block_max[b] > threshold)
                return 0;

            continue;
        }

        if (lo > threshold) {
            if ((double)block_min[b] <= threshold)
                return 0;

            metadata_in += BLOCK_VALUES;
            continue;
        }

        maybe_ids[maybe_count++] = b;

        if (!build_decisions(b, threshold))
            return 0;
    }

    out->maybe = maybe_count;
    out->truth = truth;
    out->static_bytes = maybe_count * FULL_BYTES;

    for (size_t j = 0; j < maybe_count; ++j) {
        size_t b = maybe_ids[j];
        int resolved = 0;

        for (size_t level = 0; level < LEVELS; ++level) {
            if (decisions[level][b].unknown == 0u) {
                out->exact_bytes +=
                    16u * (level + 1u);

                resolved = 1;
                break;
            }
        }

        if (!resolved)
            return 0;
    }

    for (size_t t = 0; t < TOLERANCES; ++t) {
        size_t divisor = tolerance_divisor[t];

        if (!evaluate_uniform(
                divisor, truth,
                &out->uniform[t]
            ))
            return 0;

        if (!evaluate_adaptive(
                divisor, truth,
                &out->adaptive[t]
            ))
            return 0;

        if (out->uniform[t].bytes >
                out->static_bytes ||
            out->adaptive[t].bytes >
                out->static_bytes) {

            fprintf(stderr, "FAIL: bytes exceed full scan\n");
            return 0;
        }

        if (!valid_answer(
                out->adaptive[t].lower,
                out->adaptive[t].unknown,
                truth
            )) {
            return 0;
        }
    }

    return 1;
}




static void min_median_max(
    const double values[SEEDS],
    double *minimum,
    double *median,
    double *maximum
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


static void print_summary(
    int family,
    int pass_index,
    int query_index
) {
    double exact[SEEDS];
    double uniform_1[SEEDS];
    double adaptive_1[SEEDS];
    double uniform_01[SEEDS];
    double adaptive_01[SEEDS];
    double interval_1[SEEDS];

    for (int s = 0; s < SEEDS; ++s) {
        const QueryResult *r =
            &results[family][pass_index][query_index][s];

        double baseline =
            (double)r->static_bytes;


        exact[s] = r->exact_bytes == 0u
            ? 1.0
            : baseline / (double)r->exact_bytes;

        uniform_1[s] = r->uniform[0].bytes == 0u
            ? 1.0
            : baseline / (double)r->uniform[0].bytes;

        adaptive_1[s] = r->adaptive[0].bytes == 0u
            ? 1.0
            : baseline / (double)r->adaptive[0].bytes;

        uniform_01[s] = r->uniform[1].bytes == 0u
            ? 1.0
            : baseline / (double)r->uniform[1].bytes;

        adaptive_01[s] = r->adaptive[1].bytes == 0u
            ? 1.0
            : baseline / (double)r->adaptive[1].bytes;

        interval_1[s] =
            (double)r->adaptive[0].unknown;
    }

    double lo, med, hi;
    double exact_med;
    double u1_med, a1_med;
    double u01_med, a01_med;
    double interval_med;

    min_median_max(exact, &lo, &exact_med, &hi);
    min_median_max(uniform_1, &lo, &u1_med, &hi);

    min_median_max(
        adaptive_1, &lo, &a1_med, &hi
    );

    min_median_max(
        uniform_01, &lo, &u01_med, &hi
    );

    min_median_max(
        adaptive_01, &lo, &a01_med, &hi
    );

    min_median_max(
        interval_1, &lo, &interval_med, &hi
    );

    printf(
        "sel=%2.0f%% | exact=%.2fx | "
        "1%% uniform=%.2fx adaptive=%.2fx | "
        "0.1%% uniform=%.2fx adaptive=%.2fx | "
        "adaptive 1%% width med=%.0f\n",

        100.0 * selectivities[query_index],
        exact_med,
        u1_med,
        a1_med,
        u01_med,
        a01_med,
        interval_med
    );
}




int main(void) {
    const char *family_names[FAMILIES] = {
        "temperature-like",
        "wind-like"
    };

    puts("Certified Anytime COUNT: five-seed oracle");
    puts("Truth = fully decoded STORED 8-bit ZFP data.");
    puts("COUNT interval = [lower, lower + unknown].");
    puts("1% / 0.1% tolerance uses certified lower bound.");
    puts("Savings = MAYBE-block payload bytes ONLY.");
    puts("Adaptive = offline hindsight, not a real planner.");
    puts("");

    for (int family = 0; family < FAMILIES; ++family) {
        for (int s = 0; s < SEEDS; ++s) {
            for (int p = 0; p < PASSES; ++p) {

                if (!prepare_sample(
                        family,
                        seed_values[s],
                        smooth_passes[p]
                    )) {
                    fprintf(
                        stderr,
                        "FAIL: prepare family=%d seed=%d pass=%d\n",
                        family, s, smooth_passes[p]
                    );
                    return 1;
                }

                for (int q = 0; q < QUERIES; ++q) {
                    if (!evaluate_query(
                            selectivities[q],
                            &results[family][p][q][s]
                        )) {
                        fprintf(
                            stderr,
                            "FAIL: query family=%d seed=%d "
                            "pass=%d selectivity=%.2f\n",
                            family, s,
                            smooth_passes[p],
                            selectivities[q]
                        );
                        return 1;
                    }
                }
            }
        }
    }

    for (int family = 0; family < FAMILIES; ++family) {
        printf(
            "\n========== %s ==========\n",
            family_names[family]
        );

        for (int p = 0; p < PASSES; ++p) {
            printf(
                "\n--- smoothing passes=%d ---\n",
                smooth_passes[p]
            );

            for (int q = 0; q < QUERIES; ++q)
                print_summary(family, p, q);
        }
    }

    puts("\nALL ANYTIME ORACLE CHECKS PASSED");
    return 0;
}