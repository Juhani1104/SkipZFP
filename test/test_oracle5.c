
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
    FULL_BYTES = 64,
    LEVEL_BYTES = 16,
    LEVELS = 4,

    SEEDS = 5,
    PASSES = 5,
    QUERIES = 3,
    FAMILIES = 2,
    TOLERANCES = 2,

    BATCH_SIZE = 16,

    PACKED_BYTES = VALUES + 12 + 2 * BLOCKS
};

static const double rates[LEVELS] = {2.0, 4.0, 6.0, 8.0};

static const uint32_t seeds[SEEDS] =
    {0x12345678u, 0x9e3779b9u, 0x243f6a88u, 0xb7e15162u, 0x87654321u};

static const int passes[PASSES] = {0, 1, 2, 4, 8};

static const double selectivities[QUERIES] = {0.01, 0.10, 0.50};

static const size_t divisors[TOLERANCES] = {100u, 1000u};

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
    size_t maybe;
    size_t truth;
    size_t static_bytes;
    size_t exact_bytes;

    StrategyResult uniform[TOLERANCES];
    StrategyResult online[TOLERANCES];
    StrategyResult hindsight[TOLERANCES];
} QueryResult;

static float input[VALUES];
static float noise[VALUES];
static float noise_tmp[VALUES];
static float sorted[VALUES];

static float decoded[LEVELS][BLOCKS][BLOCK_VALUES];
static float block_min[BLOCKS];
static float block_max[BLOCKS];

static double chunk_eps[LEVELS];

static unsigned char packed[PACKED_BYTES];

static QueryResult results[FAMILIES][PASSES][QUERIES][SEEDS];

static int decode_prefix(
    const unsigned char* source,
    size_t prefix_bytes,
    double rate,
    float output[BLOCK_VALUES]
) {
    unsigned char padded[FULL_BYTES] = {0};

    bitstream* bs = NULL;
    zfp_stream* zfp = NULL;
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

    double actual = zfp_stream_set_rate(zfp, rate, zfp_type_float, 3, 0);

    if (fabs(actual - rate) < 1e-9) {
        zfp_stream_rewind(zfp);

        consumed = zfp_decode_block_float_3(zfp, output);
    }

    zfp_stream_close(zfp);
    stream_close(bs);

    return consumed != 0 && consumed <= prefix_bytes * 8u;
}

static int make_noise(int family, uint32_t seed, int smooth_passes) {
    uint32_t state = seed;

    if (family == 1)
        state ^= 0xa5a5a5a5u;

    if (state == 0u)
        state = 1u;

    for (size_t i = 0; i < VALUES; ++i) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;

        noise[i] = 2.0f * (float)(state & 0x00ffffffu) / 16777215.0f - 1.0f;
    }

    for (int pass = 0; pass < smooth_passes; ++pass) {
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

    return 1;
}

static int make_sample(int family, uint32_t seed, int smooth_passes) {
    if (!make_noise(family, seed, smooth_passes))
        return 0;

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

static int prepare_sample(int family, uint32_t seed, int smooth_passes) {
    if (!make_sample(family, seed, smooth_passes))
        return 0;

    size_t written = 0;

    SzResult result = szfp_pack_chunk(
        input, SIDE, SIDE, SIDE, 8.0, 4, packed, sizeof(packed), &written
    );

    if (result != SZ_OK || written != sizeof(packed)) {
        fprintf(stderr, "FAIL: packing code=%d size=%zu\n", (int)result, written);
        return 0;
    }

    for (size_t b = 0; b < BLOCKS; ++b) {
        const unsigned char* payload = packed + b * FULL_BYTES;

        for (size_t level = 0; level < LEVELS; ++level) {

            size_t prefix_bytes = LEVEL_BYTES * (level + 1u);

            if (!decode_prefix(
                    payload, prefix_bytes, rates[level], decoded[level][b]
                )) {
                fprintf(
                    stderr,
                    "FAIL: prefix decode "
                    "block=%zu level=%zu\n",
                    b,
                    level
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

    for (size_t level = 0; level < LEVELS; ++level) {

        double maximum = 0.0;

        for (size_t b = 0; b < BLOCKS; ++b) {
            for (size_t i = 0; i < BLOCK_VALUES; ++i) {

                double error =
                    fabs((double)decoded[3][b][i] - (double)decoded[level][b][i]);

                if (error > maximum)
                    maximum = error;
            }
        }

        chunk_eps[level] = maximum == 0.0 ? 0.0 : nextafter(maximum, INFINITY);
    }

    if (chunk_eps[3] != 0.0) {
        fprintf(stderr, "FAIL: nonzero 8-bit error\n");
        return 0;
    }

    for (size_t b = 0; b < BLOCKS; ++b) {
        memcpy(sorted + b * BLOCK_VALUES, decoded[3][b], sizeof(decoded[3][b]));
    }

    qsort(sorted, VALUES, sizeof(float), compare_float);

    return 1;
}

static void metadata_interval(
    const unsigned char* offsets,
    size_t b,
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

    double lo = cmin + ((double)offsets[2u * b] / 255.0) * span - delta;

    double hi = cmin + ((double)offsets[2u * b + 1u] / 255.0) * span + delta;

    *lower = nextafter(lo, -INFINITY);
    *upper = nextafter(hi, INFINITY);
}

static Decision decide_block(size_t b, size_t level, double threshold) {
    Decision answer = {0, 0};

    for (size_t i = 0; i < BLOCK_VALUES; ++i) {

        double lower = -INFINITY;
        double upper = INFINITY;

        for (size_t l = 0; l <= level; ++l) {
            double v = (double)decoded[l][b][i];
            double eps = chunk_eps[l];

            double lo = eps == 0.0 ? v : nextafter(v - eps, -INFINITY);

            double hi = eps == 0.0 ? v : nextafter(v + eps, INFINITY);

            if (lo > lower)
                lower = lo;

            if (hi < upper)
                upper = hi;
        }

        if (lower > threshold) {
            ++answer.in;

        } else if (upper > threshold) {
            ++answer.unknown;
        }
    }

    return answer;
}

static int meets_tolerance(size_t lower, size_t unknown, size_t divisor) {
    return unknown <= lower / divisor;
}

static int valid_interval(size_t lower, size_t unknown, size_t truth) {
    return lower <= truth && unknown <= VALUES - lower && truth - lower <= unknown;
}

static int run_uniform(
    const size_t maybe_ids[BLOCKS],
    size_t maybe_count,
    size_t metadata_in,
    size_t truth,
    double threshold,
    size_t divisor,
    StrategyResult* out
) {
    memset(out, 0, sizeof(*out));

    if (maybe_count == 0u) {
        out->lower = metadata_in;
        return valid_interval(out->lower, 0u, truth);
    }

    for (size_t level = 0; level < LEVELS; ++level) {

        size_t lower = metadata_in;
        size_t unknown = 0;

        for (size_t j = 0; j < maybe_count; ++j) {
            Decision d = decide_block(maybe_ids[j], level, threshold);

            lower += d.in;
            unknown += d.unknown;
        }

        if (!valid_interval(lower, unknown, truth)) {
            fprintf(stderr, "FAIL: uniform interval\n");
            return 0;
        }

        if (meets_tolerance(lower, unknown, divisor)) {
            out->bytes = maybe_count * LEVEL_BYTES * (level + 1u);

            out->rounds = level + 1u;

            out->reads = maybe_count * (level + 1u);

            out->lower = lower;
            out->unknown = unknown;

            return 1;
        }
    }

    fprintf(stderr, "FAIL: uniform not resolved\n");
    return 0;
}

static int run_online(
    const size_t maybe_ids[BLOCKS],
    size_t maybe_count,
    size_t metadata_in,
    size_t truth,
    double threshold,
    size_t divisor,
    StrategyResult* out
) {
    size_t current_level[BLOCKS] = {0};
    Decision current[BLOCKS];

    size_t lower = metadata_in;
    size_t unknown = 0;

    memset(out, 0, sizeof(*out));

    if (maybe_count == 0u) {
        out->lower = metadata_in;
        return valid_interval(out->lower, 0u, truth);
    }

    for (size_t j = 0; j < maybe_count; ++j) {
        current[j] = decide_block(maybe_ids[j], 0u, threshold);

        lower += current[j].in;
        unknown += current[j].unknown;
    }

    out->bytes = maybe_count * LEVEL_BYTES;
    out->rounds = 1u;
    out->reads = maybe_count;

    if (!valid_interval(lower, unknown, truth)) {
        fprintf(stderr, "FAIL: online initial interval\n");
        return 0;
    }

    while (!meets_tolerance(lower, unknown, divisor)) {
        unsigned char selected[BLOCKS] = {0};
        size_t batch[BATCH_SIZE];
        size_t batch_count = 0;

        for (size_t slot = 0; slot < BATCH_SIZE; ++slot) {

            size_t best = maybe_count;

            for (size_t j = 0; j < maybe_count; ++j) {

                if (selected[j] || current_level[j] + 1u >= LEVELS ||
                    current[j].unknown == 0u) {
                    continue;
                }

                if (best == maybe_count || current[j].unknown > current[best].unknown ||
                    (current[j].unknown == current[best].unknown &&
                     maybe_ids[j] < maybe_ids[best])) {
                    best = j;
                }
            }

            if (best == maybe_count)
                break;

            selected[best] = 1u;
            batch[batch_count++] = best;
        }

        if (batch_count == 0u) {
            fprintf(stderr, "FAIL: online has no refinable block\n");
            return 0;
        }

        for (size_t k = 0; k < batch_count; ++k) {

            size_t j = batch[k];
            size_t b = maybe_ids[j];

            Decision before = current[j];

            ++current_level[j];

            Decision after = decide_block(b, current_level[j], threshold);

            if (after.in < before.in || after.unknown > before.unknown) {

                fprintf(stderr, "FAIL: online decision not monotone\n");
                return 0;
            }

            lower += after.in - before.in;
            unknown -= before.unknown - after.unknown;

            current[j] = after;
        }

        out->bytes += batch_count * LEVEL_BYTES;
        out->reads += batch_count;
        ++out->rounds;

        if (!valid_interval(lower, unknown, truth)) {
            fprintf(stderr, "FAIL: online refinement interval\n");
            return 0;
        }
    }

    out->lower = lower;
    out->unknown = unknown;

    return 1;
}

static int run_hindsight(
    const size_t maybe_ids[BLOCKS],
    size_t maybe_count,
    size_t metadata_in,
    size_t truth,
    double threshold,
    size_t divisor,
    StrategyResult* out
) {
    size_t current_level[BLOCKS] = {0};
    Decision current[BLOCKS];

    size_t lower = metadata_in;
    size_t unknown = 0;

    memset(out, 0, sizeof(*out));

    if (maybe_count == 0u) {
        out->lower = metadata_in;
        return valid_interval(out->lower, 0u, truth);
    }

    for (size_t j = 0; j < maybe_count; ++j) {
        current[j] = decide_block(maybe_ids[j], 0u, threshold);

        lower += current[j].in;
        unknown += current[j].unknown;
    }

    size_t bytes = maybe_count * LEVEL_BYTES;

    while (!meets_tolerance(lower, unknown, divisor)) {
        size_t best = maybe_count;
        size_t best_target = 0u;
        size_t best_gain = 0u;
        size_t best_steps = 1u;

        for (size_t j = 0; j < maybe_count; ++j) {

            size_t b = maybe_ids[j];
            size_t old_level = current_level[j];

            for (size_t target = old_level + 1u; target < LEVELS; ++target) {

                Decision future = decide_block(b, target, threshold);

                if (future.unknown > current[j].unknown) {
                    return 0;
                }

                size_t gain = current[j].unknown - future.unknown;

                size_t steps = target - old_level;

                if (gain == 0u)
                    continue;

                if (best == maybe_count || gain * best_steps > best_gain * steps ||
                    (gain * best_steps == best_gain * steps && steps < best_steps)) {

                    best = j;
                    best_target = target;
                    best_gain = gain;
                    best_steps = steps;
                }
            }
        }

        if (best == maybe_count) {
            fprintf(stderr, "FAIL: hindsight cannot refine\n");
            return 0;
        }

        size_t b = maybe_ids[best];

        Decision after = decide_block(b, best_target, threshold);

        lower += after.in - current[best].in;
        unknown -= current[best].unknown - after.unknown;

        bytes += LEVEL_BYTES * (best_target - current_level[best]);

        current_level[best] = best_target;
        current[best] = after;

        if (!valid_interval(lower, unknown, truth)) {
            fprintf(stderr, "FAIL: hindsight interval\n");
            return 0;
        }
    }

    out->bytes = bytes;
    out->lower = lower;
    out->unknown = unknown;

    return 1;
}

static int evaluate_query(double target_selectivity, QueryResult* out) {
    memset(out, 0, sizeof(*out));

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

    size_t desired = (size_t)llround(target_selectivity * (double)VALUES);

    if (desired < 1u)
        desired = 1u;

    if (desired >= VALUES)
        desired = VALUES - 1u;

    double threshold = (double)sorted[VALUES - desired - 1u];

    size_t truth = 0u;

    for (size_t i = 0; i < VALUES; ++i) {

        if ((double)sorted[i] > threshold)
            ++truth;
    }

    size_t maybe_ids[BLOCKS];
    size_t maybe_count = 0u;
    size_t metadata_in = 0u;

    for (size_t b = 0; b < BLOCKS; ++b) {
        double lower;
        double upper;

        metadata_interval(offsets, b, cmin, cmax, &lower, &upper);

        if ((double)block_min[b] < lower || (double)block_max[b] > upper) {

            fprintf(stderr, "FAIL: metadata misses block %zu\n", b);
            return 0;
        }

        if (upper <= threshold) {
            if ((double)block_max[b] > threshold)
                return 0;

            continue;
        }

        if (lower > threshold) {
            if ((double)block_min[b] <= threshold)
                return 0;

            metadata_in += BLOCK_VALUES;
            continue;
        }

        maybe_ids[maybe_count++] = b;
    }

    out->maybe = maybe_count;
    out->truth = truth;
    out->static_bytes = maybe_count * FULL_BYTES;

    for (size_t j = 0; j < maybe_count; ++j) {

        int resolved = 0;

        for (size_t level = 0; level < LEVELS; ++level) {

            Decision d = decide_block(maybe_ids[j], level, threshold);

            if (d.unknown == 0u) {
                out->exact_bytes += LEVEL_BYTES * (level + 1u);

                resolved = 1;
                break;
            }
        }

        if (!resolved) {
            fprintf(stderr, "FAIL: exact reference unresolved\n");
            return 0;
        }
    }

    for (size_t t = 0; t < TOLERANCES; ++t) {

        size_t divisor = divisors[t];

        if (!run_uniform(
                maybe_ids,
                maybe_count,
                metadata_in,
                truth,
                threshold,
                divisor,
                &out->uniform[t]
            )) {
            return 0;
        }

        if (!run_online(
                maybe_ids,
                maybe_count,
                metadata_in,
                truth,
                threshold,
                divisor,
                &out->online[t]
            )) {
            return 0;
        }

        if (!run_hindsight(
                maybe_ids,
                maybe_count,
                metadata_in,
                truth,
                threshold,
                divisor,
                &out->hindsight[t]
            )) {
            return 0;
        }

        if (out->uniform[t].bytes > out->static_bytes ||
            out->online[t].bytes > out->static_bytes ||
            out->hindsight[t].bytes > out->static_bytes) {

            fprintf(stderr, "FAIL: strategy fetched too many bytes\n");
            return 0;
        }
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

static double saving(size_t baseline, size_t used) {
    if (baseline == 0u || used == 0u)
        return 1.0;

    return (double)baseline / (double)used;
}

static void
print_summary(int family, int pass_index, int query_index, int tolerance_index) {
    double exact[SEEDS];
    double uniform[SEEDS];
    double online[SEEDS];
    double hindsight[SEEDS];

    double online_rounds[SEEDS];
    double online_reads[SEEDS];
    double uniform_rounds[SEEDS];

    for (int s = 0; s < SEEDS; ++s) {

        const QueryResult* r = &results[family][pass_index][query_index][s];

        const StrategyResult* u = &r->uniform[tolerance_index];

        const StrategyResult* o = &r->online[tolerance_index];

        const StrategyResult* h = &r->hindsight[tolerance_index];

        exact[s] = saving(r->static_bytes, r->exact_bytes);

        uniform[s] = saving(r->static_bytes, u->bytes);

        online[s] = saving(r->static_bytes, o->bytes);

        hindsight[s] = saving(r->static_bytes, h->bytes);

        online_rounds[s] = (double)o->rounds;

        online_reads[s] = (double)o->reads;

        uniform_rounds[s] = (double)u->rounds;
    }

    double lo, med, hi;

    double exact_med;
    double uniform_med;
    double online_med;
    double hindsight_med;

    double rounds_med;
    double rounds_min;
    double rounds_max;

    double reads_med;
    double uniform_rounds_med;

    min_median_max(exact, &lo, &exact_med, &hi);

    min_median_max(uniform, &lo, &uniform_med, &hi);

    min_median_max(online, &lo, &online_med, &hi);

    min_median_max(hindsight, &lo, &hindsight_med, &hi);

    min_median_max(online_rounds, &rounds_min, &rounds_med, &rounds_max);

    min_median_max(online_reads, &lo, &reads_med, &hi);

    min_median_max(uniform_rounds, &lo, &uniform_rounds_med, &hi);

    printf(
        "sel=%2.0f%% tol=%4.1f%% | "
        "exact=%.2fx "
        "uniform=%.2fx "
        "online=%.2fx "
        "hindsight=%.2fx | "
        "rounds online=%.0f [%.0f, %.0f] "
        "uniform=%.0f | "
        "online reads med=%.0f\n",

        100.0 * selectivities[query_index],
        tolerance_index == 0 ? 1.0 : 0.1,

        exact_med,
        uniform_med,
        online_med,
        hindsight_med,

        rounds_med,
        rounds_min,
        rounds_max,
        uniform_rounds_med,
        reads_med
    );
}

int main(void) {
    const char* names[FAMILIES] = {"temperature-like", "wind-like"};

    puts("Oracle C: causal online vs hindsight");
    puts("Exact = relative to stored 8-bit reconstruction.");
    puts("Bounds = offline observed chunk-level error.");
    puts("Savings = MAYBE payload only; NOT speedup.");
    puts("Online batch size = 16 blocks per refinement round.");
    puts("Reads = logical range reads, before coalescing.");
    puts("");

    for (int family = 0; family < FAMILIES; ++family) {

        for (int s = 0; s < SEEDS; ++s) {

            for (int p = 0; p < PASSES; ++p) {

                if (!prepare_sample(family, seeds[s], passes[p])) {

                    fprintf(
                        stderr,
                        "FAIL: prepare family=%d seed=%d "
                        "passes=%d\n",
                        family,
                        s,
                        passes[p]
                    );
                    return 1;
                }

                for (int q = 0; q < QUERIES; ++q) {

                    if (!evaluate_query(selectivities[q], &results[family][p][q][s])) {

                        fprintf(
                            stderr,
                            "FAIL: query family=%d seed=%d "
                            "passes=%d sel=%.2f\n",
                            family,
                            s,
                            passes[p],
                            selectivities[q]
                        );
                        return 1;
                    }
                }
            }
        }
    }

    for (int family = 0; family < FAMILIES; ++family) {

        printf("\n========== %s ==========\n", names[family]);

        for (int p = 0; p < PASSES; ++p) {

            printf("\n--- smoothing passes=%d ---\n", passes[p]);

            for (int q = 0; q < QUERIES; ++q) {

                for (int t = 0; t < TOLERANCES; ++t) {

                    print_summary(family, p, q, t);
                }
            }
        }
    }

    puts("\nALL ONLINE ORACLE CHECKS PASSED");
    return 0;
}