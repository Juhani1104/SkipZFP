
#define _POSIX_C_SOURCE 200809L

#define main oracle_batch_embedded_main
#include "test_oracle_batch.c"
#undef main

#include <inttypes.h>
#include <time.h>

enum {
    BENCH_PASSES = 2,
    BENCH_QUERIES = 2,
    BENCH_POLICIES = 3,
    BENCH_TRIALS = 7,
    BENCH_REPEATS = 15,
    ONLINE_BATCH = 64
};

enum {
    BENCH_FULL8,
    BENCH_B64,
    BENCH_P64
};

static const int bench_pass_indices[BENCH_PASSES] = {0, 4};
static const int bench_query_indices[BENCH_QUERIES] = {0, 2};

typedef struct {
    double lower[BLOCK_VALUES];
    double upper[BLOCK_VALUES];
    unsigned char unknown_ids[BLOCK_VALUES];
    size_t in;
    size_t unknown;
    size_t level;
} CountCache;

static CountCache caches[BLOCKS];
static volatile uint64_t bench_sink = 0;

static double measured_us
    [FAMILIES][BENCH_PASSES][BENCH_QUERIES]
    [TOLS][BENCH_POLICIES][SEEDS];

static double measured_rounds
    [FAMILIES][BENCH_PASSES][BENCH_QUERIES]
    [TOLS][BENCH_POLICIES][SEEDS];


static uint64_t cpu_clock_ns(void)
{
    struct timespec ts;

    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts) != 0) {
        perror("clock_gettime");
        return 0;
    }

    return (uint64_t)ts.tv_sec * UINT64_C(1000000000)
         + (uint64_t)ts.tv_nsec;
}


static double median7(const double values[BENCH_TRIALS])
{
    double a[BENCH_TRIALS];
    memcpy(a, values, sizeof(a));

    for (int i = 0; i < BENCH_TRIALS; ++i) {
        for (int j = i + 1; j < BENCH_TRIALS; ++j) {
            if (a[j] < a[i]) {
                double tmp = a[i];
                a[i] = a[j];
                a[j] = tmp;
            }
        }
    }

    return a[BENCH_TRIALS / 2];
}


static int update_cache(
    CountCache *c,
    const float values[BLOCK_VALUES],
    size_t level,
    double threshold,
    int first
)
{
    double eps = chunk_eps[level];

    if (first) {
        c->in = 0;
        c->unknown = 0;

        for (size_t i = 0; i < BLOCK_VALUES; ++i) {
            double v = (double)values[i];

            double lo = eps == 0.0
                ? v : nextafter(v - eps, -INFINITY);

            double hi = eps == 0.0
                ? v : nextafter(v + eps, INFINITY);

            c->lower[i] = lo;
            c->upper[i] = hi;

            if (lo > threshold) {
                ++c->in;
            } else if (hi > threshold) {
                c->unknown_ids[c->unknown++] =
                    (unsigned char)i;
            }
        }

        c->level = level;
        return 1;
    }

    size_t remaining = 0;
    size_t previous_unknown = c->unknown;

    for (size_t k = 0; k < previous_unknown; ++k) {
        size_t i = c->unknown_ids[k];
        double v = (double)values[i];

        double new_lo = eps == 0.0
            ? v : nextafter(v - eps, -INFINITY);

        double new_hi = eps == 0.0
            ? v : nextafter(v + eps, INFINITY);

        double lo = fmax(c->lower[i], new_lo);
        double hi = fmin(c->upper[i], new_hi);

        if (lo > hi)
            return 0;

        c->lower[i] = lo;
        c->upper[i] = hi;

        if (lo > threshold) {
            ++c->in;
        } else if (hi > threshold) {
            c->unknown_ids[remaining++] =
                (unsigned char)i;
        }
    }

    c->unknown = remaining;
    c->level = level;
    return 1;
}


static int decode_and_update(
    size_t block_id,
    size_t target_level,
    double threshold,
    int first
)
{
    CountCache *c = &caches[block_id];
    size_t begin = first ? 0u : c->level + 1u;
    float scratch[BLOCK_VALUES];

    if (target_level >= LEVELS || begin > target_level)
        return 0;

    for (size_t level = begin;
         level <= target_level;
         ++level) {

        if (!decode_prefix(
                packed + block_id * FULL_BYTES,
                LEVEL_BYTES * (level + 1u),
                rates[level],
                scratch
            )) {
            return 0;
        }

        if (!update_cache(
                c,
                scratch,
                level,
                threshold,
                first && level == 0u
            )) {
            return 0;
        }
    }

    return 1;
}


static int run_full8_cpu(
    const Query *q,
    StrategyResult *out
)
{
    float scratch[BLOCK_VALUES];
    size_t count = q->metadata_in;

    memset(out, 0, sizeof(*out));

    for (size_t j = 0; j < q->maybe; ++j) {
        size_t b = q->ids[j];

        if (szfp_decompress_block(
                packed + b * FULL_BYTES,
                FULL_BYTES,
                8.0,
                (int)zfp_type_float,
                3,
                scratch
            ) != SZ_OK) {
            return 0;
        }

        for (size_t i = 0; i < BLOCK_VALUES; ++i) {
            if ((double)scratch[i] > q->threshold)
                ++count;
        }
    }

    out->bytes = q->static_bytes;
    out->reads = q->maybe;
    out->rounds = q->maybe == 0u ? 0u : 1u;
    out->lower = count;
    out->unknown = 0;

    return 1;
}


static int run_incremental_cpu(
    const Query *q,
    size_t divisor,
    int predicted,
    StrategyResult *out
)
{
    size_t levels[BLOCKS] = {0};
    Decision current[BLOCKS];

    size_t lower = q->metadata_in;
    size_t unknown = 0;

    memset(out, 0, sizeof(*out));

    if (q->maybe == 0u) {
        out->lower = lower;
        return 1;
    }

    for (size_t j = 0; j < q->maybe; ++j) {
        size_t b = q->ids[j];

        levels[j] = predicted
            ? predict_initial_level(q, b, divisor)
            : 0u;

        if (!decode_and_update(
                b,
                levels[j],
                q->threshold,
                1
            )) {
            return 0;
        }

        current[j].in = caches[b].in;
        current[j].unknown = caches[b].unknown;

        lower += current[j].in;
        unknown += current[j].unknown;

        out->bytes += LEVEL_BYTES * (levels[j] + 1u);
        ++out->reads;
    }

    out->rounds = 1u;

    while (!meets_tolerance(lower, unknown, divisor)) {
        unsigned char selected[BLOCKS] = {0};
        size_t batch[ONLINE_BATCH];
        size_t count = 0;

        for (size_t slot = 0; slot < ONLINE_BATCH; ++slot) {
            size_t best = q->maybe;

            for (size_t j = 0; j < q->maybe; ++j) {
                if (selected[j] ||
                    levels[j] + 1u >= LEVELS ||
                    current[j].unknown == 0u) {
                    continue;
                }

                if (best == q->maybe ||
                    current[j].unknown > current[best].unknown ||
                    (current[j].unknown ==
                         current[best].unknown &&
                     q->ids[j] < q->ids[best])) {
                    best = j;
                }
            }

            if (best == q->maybe)
                break;

            selected[best] = 1u;
            batch[count++] = best;
        }

        if (count == 0u)
            return 0;

        for (size_t k = 0; k < count; ++k) {
            size_t j = batch[k];
            size_t b = q->ids[j];

            Decision before = current[j];
            ++levels[j];

            if (!decode_and_update(
                    b,
                    levels[j],
                    q->threshold,
                    0
                )) {
                return 0;
            }

            Decision after = {
                caches[b].in,
                caches[b].unknown
            };

            if (after.in < before.in ||
                after.unknown > before.unknown) {
                return 0;
            }

            lower += after.in - before.in;
            unknown -= before.unknown - after.unknown;
            current[j] = after;
        }

        out->bytes += count * LEVEL_BYTES;
        out->reads += count;
        ++out->rounds;
    }

    out->lower = lower;
    out->unknown = unknown;
    return 1;
}


static int run_policy(
    const Query *q,
    size_t divisor,
    int policy,
    StrategyResult *out
)
{
    if (policy == BENCH_FULL8)
        return run_full8_cpu(q, out);

    return run_incremental_cpu(
        q,
        divisor,
        policy == BENCH_P64,
        out
    );
}


static int verify_policy(
    const Query *q,
    size_t divisor,
    int policy,
    const StrategyResult references[POLICIES]
)
{
    StrategyResult actual;

    if (!run_policy(q, divisor, policy, &actual))
        return 0;

    if (!valid_interval(
            actual.lower,
            actual.unknown,
            q->truth
        ) ||
        !meets_tolerance(
            actual.lower,
            actual.unknown,
            divisor
        )) {
        return 0;
    }

    if (policy == BENCH_FULL8) {
        return actual.bytes == q->static_bytes &&
               actual.reads == q->maybe &&
               actual.rounds ==
                   (q->maybe == 0u ? 0u : 1u) &&
               actual.lower == q->truth &&
               actual.unknown == 0u;
    }

    int reference_index =
        policy == BENCH_B64
        ? POLICY_B64
        : POLICY_P64;

    const StrategyResult *expected =
        &references[reference_index];

    return actual.bytes == expected->bytes &&
           actual.reads == expected->reads &&
           actual.rounds == expected->rounds &&
           actual.lower == expected->lower &&
           actual.unknown == expected->unknown;
}


static int benchmark_policy(
    const Query *q,
    size_t divisor,
    int policy,
    double *us_per_query
)
{
    double trials[BENCH_TRIALS];

    for (int trial = 0; trial < BENCH_TRIALS; ++trial) {
        uint64_t checksum = 0;
        uint64_t start = cpu_clock_ns();

        if (start == 0u)
            return 0;

        for (int rep = 0; rep < BENCH_REPEATS; ++rep) {
            StrategyResult result;

            if (!run_policy(
                    q,
                    divisor,
                    policy,
                    &result
                )) {
                return 0;
            }

            checksum +=
                (uint64_t)result.lower +
                (uint64_t)result.unknown +
                (uint64_t)result.bytes +
                (uint64_t)result.rounds;
        }

        uint64_t end = cpu_clock_ns();

        if (end <= start)
            return 0;

        bench_sink += checksum;

        trials[trial] =
            (double)(end - start) /
            (1000.0 * BENCH_REPEATS);
    }

    *us_per_query = median7(trials);
    return 1;
}


static void print_result(
    int family,
    int pc,
    int qc,
    int tol
)
{
    double full8[SEEDS];
    double b64[SEEDS];
    double p64[SEEDS];
    double ratio_b64[SEEDS];
    double ratio_p64[SEEDS];
    double extra_p64[SEEDS];

    for (int s = 0; s < SEEDS; ++s) {
        full8[s] =
            measured_us[family][pc][qc][tol][BENCH_FULL8][s];

        b64[s] =
            measured_us[family][pc][qc][tol][BENCH_B64][s];

        p64[s] =
            measured_us[family][pc][qc][tol][BENCH_P64][s];

        ratio_b64[s] = full8[s] / b64[s];
        ratio_p64[s] = full8[s] / p64[s];
        extra_p64[s] = p64[s] - full8[s];
    }

    printf(
        "pass=%d sel=%2.0f%% tol=%3.1f%% | "
        "CPU us Full8=%.2f B64=%.2f P64=%.2f | "
        "F8/B64=%.2fx F8/P64=%.2fx | "
        "P64 extra=%.2f us\n",

        passes[bench_pass_indices[pc]],
        selectivities[bench_query_indices[qc]] * 100.0,
        tol == 0 ? 1.0 : 0.1,

        median5(full8),
        median5(b64),
        median5(p64),
        median5(ratio_b64),
        median5(ratio_p64),
        median5(extra_p64)
    );
}


int main(void)
{
    const char *family_names[FAMILIES] = {
        "temperature-like",
        "wind-like"
    };

    puts("Oracle I: local MAYBE-stage query CPU");
    puts("Measured: decode + COUNT + scheduling.");
    puts("Excludes: I/O, ingest, metadata filtering and setup.");
    puts("Reference: fully decoded STORED 8-bit values.");
    puts("");

    for (int family = 0; family < FAMILIES; ++family) {
        for (int s = 0; s < SEEDS; ++s) {
            for (int pc = 0; pc < BENCH_PASSES; ++pc) {
                int pass_index = bench_pass_indices[pc];

                if (!prepare_sample(
                        family,
                        seeds[s],
                        passes[pass_index]
                    )) {
                    fprintf(stderr, "FAIL: prepare sample\n");
                    return 1;
                }

                for (int qc = 0; qc < BENCH_QUERIES; ++qc) {
                    int query_index = bench_query_indices[qc];
                    Query q;

                    if (!build_query(
                            selectivities[query_index],
                            &q
                        )) {
                        fprintf(stderr, "FAIL: build query\n");
                        return 1;
                    }

                    for (int tol = 0; tol < TOLS; ++tol) {
                        StrategyResult references[POLICIES];

                        if (!evaluate_policies(
                                &q,
                                divisors[tol],
                                references
                            )) {
                            fprintf(stderr, "FAIL: references\n");
                            return 1;
                        }

                        for (int policy = 0;
                             policy < BENCH_POLICIES;
                             ++policy) {

                            if (!verify_policy(
                                    &q,
                                    divisors[tol],
                                    policy,
                                    references
                                )) {
                                fprintf(
                                    stderr,
                                    "FAIL: policy mismatch "
                                    "family=%d seed=%d pass=%d "
                                    "sel=%.2f tol=%zu policy=%d\n",
                                    family,
                                    s,
                                    passes[pass_index],
                                    selectivities[query_index],
                                    divisors[tol],
                                    policy
                                );
                                return 1;
                            }
                        }

                        double times[BENCH_POLICIES] = {0};

                        /*
                         * Rotate measurement order across
                         * seeds to reduce order effects.
                         */
                        for (int slot = 0;
                             slot < BENCH_POLICIES;
                             ++slot) {

                            int policy =
                                (slot + s) % BENCH_POLICIES;

                            if (!benchmark_policy(
                                    &q,
                                    divisors[tol],
                                    policy,
                                    &times[policy]
                                )) {
                                fprintf(
                                    stderr,
                                    "FAIL: benchmark policy=%d\n",
                                    policy
                                );
                                return 1;
                            }
                        }

                        for (int policy = 0;
                             policy < BENCH_POLICIES;
                             ++policy) {

                            measured_us[family]
                                       [pc]
                                       [qc]
                                       [tol]
                                       [policy]
                                       [s] = times[policy];

                            measured_rounds[family]
                                           [pc]
                                           [qc]
                                           [tol]
                                           [policy]
                                           [s] =
                                policy == BENCH_FULL8
                                ? (double)(
                                    q.maybe == 0u ? 0u : 1u
                                  )
                                : (double)references[
                                    policy == BENCH_B64
                                    ? POLICY_B64
                                    : POLICY_P64
                                  ].rounds;
                        }
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

        for (int pc = 0; pc < BENCH_PASSES; ++pc) {
            for (int qc = 0; qc < BENCH_QUERIES; ++qc) {
                for (int tol = 0; tol < TOLS; ++tol) {
                    print_result(family, pc, qc, tol);
                }
            }
        }
    }

    printf(
        "\nBenchmark checksum: %" PRIu64 "\n",
        (uint64_t)bench_sink
    );

    puts("ALL LOCAL QUERY CPU CHECKS PASSED");
    return 0;
}