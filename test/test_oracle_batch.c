
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
    LEVEL_BYTES = 16,
    LEVELS = 4,

    SEEDS = 5,
    PASSES = 5,
    QUERIES = 3,
    FAMILIES = 2,
    TOLS = 2,

    POLICIES = 6,
    MAX_BATCH = 128,

    PACKED_BYTES = VALUES + 12 + 2 * BLOCKS
};

enum {
    POLICY_UNIFORM,
    POLICY_B16,
    POLICY_B32,
    POLICY_B64,
    POLICY_B128,
    POLICY_P64
};

static const double rates[LEVELS] = {
    2.0, 4.0, 6.0, 8.0
};

static const uint32_t seeds[SEEDS] = {
    0x12345678u,
    0x9e3779b9u,
    0x243f6a88u,
    0xb7e15162u,
    0x87654321u
};

static const int passes[PASSES] = {
    0, 1, 2, 4, 8
};

static const double selectivities[QUERIES] = {
    0.01, 0.10, 0.50
};


static const size_t divisors[TOLS] = {
    100u, 1000u
};

static const char *policy_names[POLICIES] = {
    "U", "B16", "B32", "B64", "B128", "P64"
};

typedef struct {
    size_t in;
    size_t unknown;
} Decision;

typedef struct {
    size_t bytes;
    size_t rounds;
    size_t reads;
    size_t lower;
    size_t unknown;
} StrategyResult;

typedef struct {
    double threshold;
    double lo[BLOCKS];
    double hi[BLOCKS];

    size_t ids[BLOCKS];
    size_t maybe;
    size_t metadata_in;
    size_t truth;
    size_t static_bytes;


    double estimated_count;
} Query;

static float input[VALUES];
static float noise[VALUES];
static float noise_tmp[VALUES];
static float sorted[VALUES];

static float decoded[LEVELS][BLOCKS][BLOCK_VALUES];
static float block_min[BLOCKS];
static float block_max[BLOCKS];

static double chunk_eps[LEVELS];
static unsigned char packed[PACKED_BYTES];

static StrategyResult
    results[FAMILIES][PASSES][QUERIES][TOLS][POLICIES][SEEDS];

static size_t
    baseline_bytes[FAMILIES][PASSES][QUERIES][SEEDS];




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

    if (prefix_bytes == 0 ||
        prefix_bytes > FULL_BYTES)
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

        consumed = zfp_decode_block_float_3(
            zfp, output
        );
    }

    zfp_stream_close(zfp);
    stream_close(bs);

    return consumed != 0 &&
           consumed <= prefix_bytes * 8u;
}




static int make_noise(
    int family,
    uint32_t seed,
    int smooth_passes
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


    for (int pass = 0; pass < smooth_passes; ++pass) {
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
                                    SIDE * (
                                        ys[dy] + SIDE * zs[dz]
                                    );

                                sum += noise[j];
                            }
                        }
                    }

                    size_t i =
                        x + SIDE * (y + SIDE * z);

                    noise_tmp[i] = (float)(sum / 27.0);
                }
            }
        }

        memcpy(noise, noise_tmp, sizeof(noise));
    }

    /*
     * Normalize after smoothing:
     * mean ~= 0, standard deviation ~= 1.
     */
    double mean = 0.0;
    double variance = 0.0;

    for (size_t i = 0; i < VALUES; ++i)
        mean += (double)noise[i];

    mean /= (double)VALUES;

    for (size_t i = 0; i < VALUES; ++i) {
        double d = (double)noise[i] - mean;
        variance += d * d;
    }

    double stddev = sqrt(
        variance / (double)VALUES
    );

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
    int smooth_passes
) {
    if (!make_noise(family, seed, smooth_passes))
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
                0.15f * sinf(
                    0.3f * x + 0.2f * y
                );

            input[i] = base + 0.8f * noise[i];

        } else {
            float base =
                6.0f * sinf(
                    0.23f * x + 0.11f * z
                ) -
                4.0f * cosf(0.19f * y) +
                0.07f * z;

            input[i] = base + 1.5f * noise[i];
        }
    }

    return 1;
}


static int compare_float(
    const void *a,
    const void *b
) {
    float x = *(const float *)a;
    float y = *(const float *)b;

    return (x > y) - (x < y);
}




static int prepare_sample(
    int family,
    uint32_t seed,
    int smooth_passes
) {
    if (!make_sample(family, seed, smooth_passes))
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

    if (result != SZ_OK ||
        written != sizeof(packed)) {

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
            size_t prefix_bytes =
                LEVEL_BYTES * (level + 1u);

            if (!decode_prefix(
                    payload,
                    prefix_bytes,
                    rates[level],
                    decoded[level][b]
                )) {

                fprintf(
                    stderr,
                    "FAIL: prefix decode "
                    "block=%zu level=%zu\n",
                    b, level
                );
                return 0;
            }

            for (size_t i = 0; i < BLOCK_VALUES; ++i) {
                if (!isfinite(decoded[level][b][i])) {
                    fprintf(
                        stderr,
                        "FAIL: non-finite reconstruction\n"
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

    if (chunk_eps[3] != 0.0) {
        fprintf(stderr, "FAIL: nonzero 8-bit error\n");
        return 0;
    }

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
        ((double)offsets[2u * b] / 255.0)
            * span -
        delta;

    double hi =
        cmin +
        ((double)offsets[2u * b + 1u] / 255.0)
            * span +
        delta;

    *lower = nextafter(lo, -INFINITY);
    *upper = nextafter(hi, INFINITY);
}




static int build_query(
    double target_selectivity,
    Query *q
) {
    memset(q, 0, sizeof(*q));

    size_t desired = (size_t)llround(
        target_selectivity * (double)VALUES
    );

    if (desired < 1u)
        desired = 1u;

    if (desired >= VALUES)
        desired = VALUES - 1u;

    q->threshold =
        (double)sorted[VALUES - desired - 1u];

    for (size_t i = 0; i < VALUES; ++i) {
        if ((double)sorted[i] > q->threshold)
            ++q->truth;
    }

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

    for (size_t b = 0; b < BLOCKS; ++b) {
        metadata_interval(
            offsets,
            b,
            cmin,
            cmax,
            &q->lo[b],
            &q->hi[b]
        );

        /* Ground-truth-based safety check ONLY. */
        if ((double)block_min[b] < q->lo[b] ||
            (double)block_max[b] > q->hi[b]) {

            fprintf(
                stderr,
                "FAIL: metadata misses block %zu\n",
                b
            );
            return 0;
        }

        if (q->hi[b] <= q->threshold) {
            if ((double)block_max[b] > q->threshold)
                return 0;

            continue;
        }

        if (q->lo[b] > q->threshold) {
            if ((double)block_min[b] <= q->threshold)
                return 0;

            q->metadata_in += BLOCK_VALUES;
            continue;
        }

        q->ids[q->maybe++] = b;
    }

    q->static_bytes = q->maybe * FULL_BYTES;


    q->estimated_count = (double)q->metadata_in;

    for (size_t j = 0; j < q->maybe; ++j) {
        size_t b = q->ids[j];

        double width = q->hi[b] - q->lo[b];
        double probability;

        if (!(width > 0.0)) {
            probability =
                q->lo[b] > q->threshold ? 1.0 : 0.0;
        } else {
            probability =
                (q->hi[b] - q->threshold) / width;

            if (probability < 0.0)
                probability = 0.0;

            if (probability > 1.0)
                probability = 1.0;
        }

        q->estimated_count +=
            BLOCK_VALUES * probability;
    }

    return 1;
}




static Decision decide_block(
    size_t b,
    size_t level,
    double threshold
) {
    Decision d = {0, 0};

    for (size_t i = 0; i < BLOCK_VALUES; ++i) {
        double lower = -INFINITY;
        double upper = INFINITY;

        for (size_t l = 0; l <= level; ++l) {
            double v = (double)decoded[l][b][i];
            double eps = chunk_eps[l];

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
        }

        if (lower > threshold) {
            ++d.in;

        } else if (upper > threshold) {
            ++d.unknown;
        }
    }

    return d;
}




static int valid_interval(
    size_t lower,
    size_t unknown,
    size_t truth
) {
    return lower <= truth &&
           unknown <= VALUES - lower &&
           truth - lower <= unknown;
}


static int meets_tolerance(
    size_t lower,
    size_t unknown,
    size_t divisor
) {
    return unknown <= lower / divisor;
}




static int run_uniform(
    const Query *q,
    size_t divisor,
    StrategyResult *out
) {
    memset(out, 0, sizeof(*out));

    if (q->maybe == 0u) {
        out->lower = q->metadata_in;
        return valid_interval(out->lower, 0u, q->truth);
    }

    for (size_t level = 0; level < LEVELS; ++level) {
        size_t lower = q->metadata_in;
        size_t unknown = 0;

        for (size_t j = 0; j < q->maybe; ++j) {
            Decision d = decide_block(
                q->ids[j],
                level,
                q->threshold
            );

            lower += d.in;
            unknown += d.unknown;
        }

        if (!valid_interval(lower, unknown, q->truth)) {
            fprintf(stderr, "FAIL: uniform interval\n");
            return 0;
        }

        if (meets_tolerance(lower, unknown, divisor)) {
            out->bytes =
                q->maybe *
                LEVEL_BYTES *
                (level + 1u);

            out->rounds = level + 1u;

            out->reads =
                q->maybe * (level + 1u);

            out->lower = lower;
            out->unknown = unknown;

            return 1;
        }
    }

    fprintf(stderr, "FAIL: uniform not resolved\n");
    return 0;
}




static size_t predict_initial_level(
    const Query *q,
    size_t b,
    size_t divisor
) {
    double width = q->hi[b] - q->lo[b];


    double target_unknown =
        (q->estimated_count / (double)divisor) /
        (double)q->maybe;

    if (!(width > 0.0))
        return 2u;

    for (size_t level = 0; level < 3u; ++level) {
        double eps = chunk_eps[level];

        double uncertain_lo =
            q->threshold - eps;

        double uncertain_hi =
            q->threshold + eps;

        double overlap_lo =
            fmax(q->lo[b], uncertain_lo);

        double overlap_hi =
            fmin(q->hi[b], uncertain_hi);

        double overlap =
            fmax(0.0, overlap_hi - overlap_lo);

        double expected_unknown =
            BLOCK_VALUES * overlap / width;

        if (expected_unknown <= target_unknown)
            return level;
    }

    return 2u;
}




static int run_online(
    const Query *q,
    size_t divisor,
    size_t batch_size,
    int pred,
    StrategyResult *out
) {
    size_t level[BLOCKS] = {0};
    Decision current[BLOCKS];

    size_t lower = q->metadata_in;
    size_t unknown = 0;

    memset(out, 0, sizeof(*out));

    if (q->maybe == 0u) {
        out->lower = lower;
        return valid_interval(lower, 0u, q->truth);
    }


    for (size_t j = 0; j < q->maybe; ++j) {
        size_t b = q->ids[j];

        level[j] = pred
            ? predict_initial_level(q, b, divisor)
            : 0u;

        current[j] = decide_block(
            b,
            level[j],
            q->threshold
        );

        lower += current[j].in;
        unknown += current[j].unknown;

        out->bytes +=
            LEVEL_BYTES * (level[j] + 1u);

        ++out->reads;
    }

    out->rounds = 1u;

    if (!valid_interval(lower, unknown, q->truth)) {
        fprintf(stderr, "FAIL: initial online interval\n");
        return 0;
    }


    while (!meets_tolerance(
        lower, unknown, divisor
    )) {
        unsigned char selected[BLOCKS] = {0};
        size_t batch[MAX_BATCH];
        size_t n = 0;


        for (size_t slot = 0;
             slot < batch_size;
             ++slot) {

            size_t best = q->maybe;

            for (size_t j = 0; j < q->maybe; ++j) {
                if (selected[j] ||
                    level[j] + 1u >= LEVELS ||
                    current[j].unknown == 0u) {
                    continue;
                }

                if (best == q->maybe ||
                    current[j].unknown >
                        current[best].unknown ||
                    (current[j].unknown ==
                        current[best].unknown &&
                     q->ids[j] < q->ids[best])) {

                    best = j;
                }
            }

            if (best == q->maybe)
                break;

            selected[best] = 1u;
            batch[n++] = best;
        }

        if (n == 0u) {
            fprintf(
                stderr,
                "FAIL: no refinable block remains\n"
            );
            return 0;
        }


        for (size_t k = 0; k < n; ++k) {
            size_t j = batch[k];
            size_t b = q->ids[j];

            Decision before = current[j];

            ++level[j];


            Decision after = decide_block(
                b,
                level[j],
                q->threshold
            );

            if (after.in < before.in ||
                after.unknown > before.unknown) {

                fprintf(
                    stderr,
                    "FAIL: nonmonotone online decision\n"
                );
                return 0;
            }

            lower += after.in - before.in;

            unknown -=
                before.unknown - after.unknown;

            current[j] = after;
        }

        out->bytes += n * LEVEL_BYTES;
        out->reads += n;
        ++out->rounds;

        if (!valid_interval(lower, unknown, q->truth)) {
            fprintf(
                stderr,
                "FAIL: refinement COUNT interval\n"
            );
            return 0;
        }
    }

    out->lower = lower;
    out->unknown = unknown;

    if (out->bytes > q->static_bytes) {
        fprintf(
            stderr,
            "FAIL: fetched more than full MAYBE payload\n"
        );
        return 0;
    }

    return 1;
}




static int evaluate_policies(
    const Query *q,
    size_t divisor,
    StrategyResult out[POLICIES]
) {
    if (!run_uniform(
            q, divisor,
            &out[POLICY_UNIFORM]
        ))
        return 0;

    const size_t batches[4] = {
        16u, 32u, 64u, 128u
    };

    for (size_t i = 0; i < 4u; ++i) {
        if (!run_online(
                q,
                divisor,
                batches[i],
                0,
                &out[POLICY_B16 + i]
            ))
            return 0;
    }

    if (!run_online(
            q,
            divisor,
            64u,
            1,
            &out[POLICY_P64]
        ))
        return 0;

    for (size_t p = 0; p < POLICIES; ++p) {
        if (!valid_interval(
                out[p].lower,
                out[p].unknown,
                q->truth
            )) {
            fprintf(stderr, "FAIL: policy COUNT interval\n");
            return 0;
        }

        if (!meets_tolerance(
                out[p].lower,
                out[p].unknown,
                divisor
            )) {
            fprintf(stderr, "FAIL: tolerance not met\n");
            return 0;
        }

        if (out[p].bytes > q->static_bytes) {
            fprintf(stderr, "FAIL: excessive payload\n");
            return 0;
        }
    }

    return 1;
}




static double median5(
    const double values[SEEDS]
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

    return tmp[SEEDS / 2];
}


static void print_summary(
    int family,
    int pass_index,
    int query_index,
    int tol_index
) {
    printf(
        "sel=%2.0f%% tol=%3.1f%% | ",
        selectivities[query_index] * 100.0,
        tol_index == 0 ? 1.0 : 0.1
    );

    for (int policy = 0; policy < POLICIES; ++policy) {
        double saving[SEEDS];
        double rounds[SEEDS];
        double reads[SEEDS];

        for (int s = 0; s < SEEDS; ++s) {
            const StrategyResult *r =
                &results[family]
                        [pass_index]
                        [query_index]
                        [tol_index]
                        [policy]
                        [s];

            size_t baseline =
                baseline_bytes[family]
                              [pass_index]
                              [query_index]
                              [s];

            saving[s] =
                baseline == 0u || r->bytes == 0u
                ? 1.0
                : (double)baseline /
                  (double)r->bytes;

            rounds[s] = (double)r->rounds;
            reads[s] = (double)r->reads;
        }

        printf(
            "%s:x%.2f/r%.0f/q%.0f%s",
            policy_names[policy],
            median5(saving),
            median5(rounds),
            median5(reads),
            policy == POLICIES - 1 ? "" : " | "
        );
    }

    putchar('\n');
}




int main(void) {
    const char *names[FAMILIES] = {
        "temperature-like",
        "wind-like"
    };

    puts("Oracle D: batch size + first-round prediction");
    puts("Reference: fully decoded STORED 8-bit ZFP data.");
    puts("Bounds: offline observed chunk-level error.");
    puts("x = MAYBE payload saving, NOT speedup.");
    puts("r = dependent fetch phases.");
    puts("q = logical block reads, before coalescing.");
    puts("P64 = predicted first prefix + batch-64 refinement.");
    puts("");

    for (int family = 0;
         family < FAMILIES;
         ++family) {

        for (int s = 0; s < SEEDS; ++s) {
            for (int p = 0; p < PASSES; ++p) {
                if (!prepare_sample(
                        family,
                        seeds[s],
                        passes[p]
                    )) {

                    fprintf(
                        stderr,
                        "FAIL: prepare family=%d seed=%d pass=%d\n",
                        family, s, passes[p]
                    );
                    return 1;
                }

                for (int qidx = 0;
                     qidx < QUERIES;
                     ++qidx) {

                    Query q;

                    if (!build_query(
                            selectivities[qidx],
                            &q
                        )) {

                        fprintf(
                            stderr,
                            "FAIL: build query\n"
                        );
                        return 1;
                    }

                    baseline_bytes[family]
                                  [p]
                                  [qidx]
                                  [s] =
                        q.static_bytes;

                    for (int tol = 0;
                         tol < TOLS;
                         ++tol) {

                        if (!evaluate_policies(
                                &q,
                                divisors[tol],
                                results[family]
                                       [p]
                                       [qidx]
                                       [tol]
                                       [0]
                                       + s
                            )) {

                            return 1;
                        }
                    }
                }
            }
        }
    }

    for (int family = 0; family < FAMILIES; ++family) {
        printf(
            "\n========== %s ==========\n",
            names[family]
        );

        for (int p = 0; p < PASSES; ++p) {
            printf(
                "\n--- smoothing passes=%d ---\n",
                passes[p]
            );

            for (int q = 0; q < QUERIES; ++q) {
                for (int tol = 0; tol < TOLS; ++tol) {
                    print_summary(
                        family, p, q, tol
                    );
                }
            }
        }
    }

    puts("\nALL BATCH ORACLE CHECKS PASSED");
    return 0;
}