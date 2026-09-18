
#define _POSIX_C_SOURCE 200809L

#define main oracle_batch_embedded_main
#include "test_oracle_batch.c"
#undef main

#include <inttypes.h>
#include <time.h>

enum {
    JOINT_PASSES = 2,
    JOINT_QUERIES = 2,
    JOINT_POLICIES = 3,
    JOINT_TRIALS = 7,
    JOINT_REPEATS = 15,
    JOINT_MAX_ROUNDS = 1 + 3 * BLOCKS
};

enum {
    JOINT_FULL8,
    JOINT_B64,
    JOINT_P64
};

static const int joint_pass_indices[JOINT_PASSES] = {0, 4};
static const int joint_query_indices[JOINT_QUERIES] = {0, 2};

static const char *joint_policy_names[JOINT_POLICIES] = {
    "Full8", "B64", "P64"
};

typedef struct {
    double lower[BLOCK_VALUES];
    double upper[BLOCK_VALUES];
    unsigned char unknown_ids[BLOCK_VALUES];
    size_t in;
    size_t unknown;
    size_t level;
} JointCache;

typedef struct {
    StrategyResult result;
    size_t count;
    size_t bytes[JOINT_MAX_ROUNDS];
    size_t reads[JOINT_MAX_ROUNDS];
} JointTrace;

typedef struct {
    StrategyResult result;
    double cpu_us;
    size_t waves;
} JointSample;

typedef struct {
    const char *name;
    double latency_ms;
    double bandwidth_mb_s;
    double wave_ms;
    size_t parallelism;
} JointScenario;

static const JointScenario joint_scenarios[] = {
    {"LOW_RTT_SLOW_LINK",  0.05,  0.25, 0.02, 64},
    {"LOW_RTT_FAST_LINK",  0.05, 50.00, 0.02, 64},
    {"MID_RTT_FAST_LINK",  5.00, 50.00, 0.02, 64},
    {"HIGH_RTT_FAST_LINK", 50.0, 50.00, 0.02, 64}
};

enum {
    JOINT_SCENARIOS =
        sizeof(joint_scenarios) / sizeof(joint_scenarios[0])
};

static JointCache joint_cache[BLOCKS];

static JointSample joint_samples
    [FAMILIES]
    [JOINT_PASSES]
    [JOINT_QUERIES]
    [TOLS]
    [JOINT_POLICIES]
    [SEEDS];

static volatile uint64_t joint_sink = 0;


static uint64_t joint_clock_ns(void)
{
    struct timespec ts;

    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts) != 0) {
        perror("clock_gettime");
        return 0;
    }

    return (uint64_t)ts.tv_sec * UINT64_C(1000000000)
         + (uint64_t)ts.tv_nsec;
}


static double joint_median(double *values, size_t n)
{
    double copy[SEEDS > JOINT_TRIALS ? SEEDS : JOINT_TRIALS];

    if (n == 0 || n > sizeof(copy) / sizeof(copy[0]))
        return NAN;

    memcpy(copy, values, n * sizeof(double));

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            if (copy[j] < copy[i]) {
                double tmp = copy[i];
                copy[i] = copy[j];
                copy[j] = tmp;
            }
        }
    }

    return copy[n / 2];
}


static int joint_record(
    JointTrace *trace,
    size_t bytes,
    size_t reads
)
{
    if (trace->count >= JOINT_MAX_ROUNDS ||
        bytes == 0 || reads == 0)
        return 0;

    size_t k = trace->count++;

    trace->bytes[k] = bytes;
    trace->reads[k] = reads;

    trace->result.bytes += bytes;
    trace->result.reads += reads;
    ++trace->result.rounds;

    return 1;
}


static int joint_update(
    JointCache *c,
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


static int joint_decode_update(
    size_t block_id,
    size_t target_level,
    double threshold,
    int first
)
{
    JointCache *c = &joint_cache[block_id];
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
            ))
            return 0;

        if (!joint_update(
                c,
                scratch,
                level,
                threshold,
                first && level == 0u
            ))
            return 0;
    }

    return 1;
}


static int joint_full8(
    const Query *q,
    JointTrace *trace
)
{
    float scratch[BLOCK_VALUES];
    size_t count = q->metadata_in;

    memset(trace, 0, sizeof(*trace));

    for (size_t j = 0; j < q->maybe; ++j) {
        size_t b = q->ids[j];

        if (szfp_decompress_block(
                packed + b * FULL_BYTES,
                FULL_BYTES,
                8.0,
                (int)zfp_type_float,
                3,
                scratch
            ) != SZ_OK)
            return 0;

        for (size_t i = 0; i < BLOCK_VALUES; ++i) {
            if ((double)scratch[i] > q->threshold)
                ++count;
        }
    }

    if (q->maybe != 0 &&
        !joint_record(trace, q->static_bytes, q->maybe))
        return 0;

    trace->result.lower = count;
    trace->result.unknown = 0;

    return 1;
}


static int joint_online(
    const Query *q,
    size_t divisor,
    int predicted,
    JointTrace *trace
)
{
    size_t levels[BLOCKS] = {0};
    Decision current[BLOCKS];

    size_t lower = q->metadata_in;
    size_t unknown = 0;
    size_t first_bytes = 0;

    memset(trace, 0, sizeof(*trace));

    if (q->maybe == 0) {
        trace->result.lower = lower;
        return 1;
    }

    for (size_t j = 0; j < q->maybe; ++j) {
        size_t b = q->ids[j];

        levels[j] = predicted
            ? predict_initial_level(q, b, divisor)
            : 0u;

        if (!joint_decode_update(
                b,
                levels[j],
                q->threshold,
                1
            ))
            return 0;

        current[j].in = joint_cache[b].in;
        current[j].unknown = joint_cache[b].unknown;

        lower += current[j].in;
        unknown += current[j].unknown;

        first_bytes += LEVEL_BYTES * (levels[j] + 1u);
    }

    if (!joint_record(trace, first_bytes, q->maybe))
        return 0;

    while (!meets_tolerance(lower, unknown, divisor)) {
        unsigned char selected[BLOCKS] = {0};
        size_t batch[64];
        size_t n = 0;

        for (size_t slot = 0; slot < 64u; ++slot) {
            size_t best = q->maybe;

            for (size_t j = 0; j < q->maybe; ++j) {
                if (selected[j] ||
                    levels[j] + 1u >= LEVELS ||
                    current[j].unknown == 0u)
                    continue;

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

        if (n == 0)
            return 0;

        for (size_t k = 0; k < n; ++k) {
            size_t j = batch[k];
            size_t b = q->ids[j];

            Decision before = current[j];
            ++levels[j];

            if (!joint_decode_update(
                    b,
                    levels[j],
                    q->threshold,
                    0
                ))
                return 0;

            Decision after = {
                joint_cache[b].in,
                joint_cache[b].unknown
            };

            if (after.in < before.in ||
                after.unknown > before.unknown)
                return 0;

            lower += after.in - before.in;
            unknown -= before.unknown - after.unknown;

            current[j] = after;
        }

        if (!joint_record(trace, n * LEVEL_BYTES, n))
            return 0;
    }

    trace->result.lower = lower;
    trace->result.unknown = unknown;

    return 1;
}


static int joint_execute(
    const Query *q,
    size_t divisor,
    int policy,
    JointTrace *trace
)
{
    if (policy == JOINT_FULL8)
        return joint_full8(q, trace);

    return joint_online(
        q,
        divisor,
        policy == JOINT_P64,
        trace
    );
}


static int joint_check(
    const Query *q,
    size_t divisor,
    int policy,
    const JointTrace *trace,
    const StrategyResult references[POLICIES]
)
{
    const StrategyResult *r = &trace->result;

    size_t bytes = 0;
    size_t reads = 0;

    for (size_t k = 0; k < trace->count; ++k) {
        bytes += trace->bytes[k];
        reads += trace->reads[k];
    }

    if (bytes != r->bytes ||
        reads != r->reads ||
        trace->count != r->rounds ||
        !valid_interval(r->lower, r->unknown, q->truth) ||
        !meets_tolerance(r->lower, r->unknown, divisor))
        return 0;

    if (policy == JOINT_FULL8) {
        return r->bytes == q->static_bytes &&
               r->reads == q->maybe &&
               r->rounds == (q->maybe == 0u ? 0u : 1u) &&
               r->lower == q->truth &&
               r->unknown == 0u;
    }

    const StrategyResult *expected =
        &references[
            policy == JOINT_B64
            ? POLICY_B64
            : POLICY_P64
        ];

    return r->bytes == expected->bytes &&
           r->reads == expected->reads &&
           r->rounds == expected->rounds &&
           r->lower == expected->lower &&
           r->unknown == expected->unknown;
}


static size_t joint_waves(
    const JointTrace *trace,
    size_t parallelism
)
{
    size_t total = 0;

    for (size_t k = 0; k < trace->count; ++k) {
        size_t reads = trace->reads[k];

        total += reads / parallelism +
                 (reads % parallelism != 0u);
    }

    return total;
}


static int joint_benchmark(
    const Query *q,
    size_t divisor,
    int policy,
    double *cpu_us
)
{
    double trials[JOINT_TRIALS];

    for (int trial = 0; trial < JOINT_TRIALS; ++trial) {
        uint64_t checksum = 0;
        uint64_t start = joint_clock_ns();

        if (start == 0u)
            return 0;

        for (int rep = 0; rep < JOINT_REPEATS; ++rep) {
            JointTrace trace;

            if (!joint_execute(
                    q,
                    divisor,
                    policy,
                    &trace
                ))
                return 0;

            checksum +=
                (uint64_t)trace.result.lower +
                (uint64_t)trace.result.unknown +
                (uint64_t)trace.result.bytes +
                (uint64_t)trace.result.reads +
                (uint64_t)trace.result.rounds;
        }

        uint64_t end = joint_clock_ns();

        if (end <= start)
            return 0;

        joint_sink += checksum;

        trials[trial] =
            (double)(end - start) /
            (1000.0 * JOINT_REPEATS);
    }

    *cpu_us = joint_median(trials, JOINT_TRIALS);

    return isfinite(*cpu_us) && *cpu_us >= 0.0;
}


static double joint_model_ms(
    const JointSample *sample,
    const JointScenario *scenario
)
{
    const StrategyResult *r = &sample->result;

    double transfer_ms =
        (double)r->bytes /
        (1000.0 * scenario->bandwidth_mb_s);

    return
        (double)r->rounds * scenario->latency_ms +
        (double)sample->waves * scenario->wave_ms +
        transfer_ms +
        sample->cpu_us / 1000.0;
}


static void joint_print_case(
    int scenario_index,
    int family,
    int pc,
    int qc,
    int tol
)
{
    const JointScenario *scenario =
        &joint_scenarios[scenario_index];

    double times[JOINT_POLICIES][SEEDS];
    double ratios_b64[SEEDS];
    double ratios_p64[SEEDS];
    double delta_b64[SEEDS];
    double delta_p64[SEEDS];

    int wins_b64 = 0;
    int wins_p64 = 0;

    for (int s = 0; s < SEEDS; ++s) {
        for (int policy = 0;
             policy < JOINT_POLICIES;
             ++policy) {

            times[policy][s] = joint_model_ms(
                &joint_samples[family][pc][qc][tol][policy][s],
                scenario
            );
        }

        double full8 = times[JOINT_FULL8][s];
        double b64 = times[JOINT_B64][s];
        double p64 = times[JOINT_P64][s];

        if (b64 <= 0.0 || p64 <= 0.0) {
            fprintf(stderr, "FAIL: nonpositive model time\n");
            exit(EXIT_FAILURE);
        }

        ratios_b64[s] = full8 / b64;
        ratios_p64[s] = full8 / p64;

        delta_b64[s] = b64 - full8;
        delta_p64[s] = p64 - full8;

        if (b64 < full8)
            ++wins_b64;

        if (p64 < full8)
            ++wins_p64;
    }

    printf(
        "pass=%d sel=%2.0f%% tol=%3.1f%% | "
        "model ms F8=%.3f B64=%.3f P64=%.3f | "
        "F8/B64=%.2fx F8/P64=%.2fx | "
        "delta B64/P64=%+.3f/%+.3f ms | "
        "wins B64/P64=%d/%d\n",

        passes[joint_pass_indices[pc]],
        100.0 * selectivities[joint_query_indices[qc]],
        tol == 0 ? 1.0 : 0.1,

        joint_median(times[JOINT_FULL8], SEEDS),
        joint_median(times[JOINT_B64], SEEDS),
        joint_median(times[JOINT_P64], SEEDS),

        joint_median(ratios_b64, SEEDS),
        joint_median(ratios_p64, SEEDS),

        joint_median(delta_b64, SEEDS),
        joint_median(delta_p64, SEEDS),

        wins_b64,
        wins_p64
    );
}


static void joint_print_break_even(
    int family,
    int tol,
    int policy,
    double bandwidth
)
{
    const int pc = 1;
    const int qc = 1;

    double cutoffs[SEEDS];
    size_t cutoff_count = 0;

    int always = 0;
    int never = 0;

    for (int s = 0; s < SEEDS; ++s) {
        const JointSample *f =
            &joint_samples[family][pc][qc][tol][JOINT_FULL8][s];

        const JointSample *p =
            &joint_samples[family][pc][qc][tol][policy][s];

        long long extra_rounds =
            (long long)p->result.rounds -
            (long long)f->result.rounds;

        double delta_at_zero =
            ((double)p->waves - (double)f->waves) * 0.02 +
            ((double)p->result.bytes -
             (double)f->result.bytes) /
                (1000.0 * bandwidth) +
            (p->cpu_us - f->cpu_us) / 1000.0;

        if (extra_rounds == 0) {
            if (delta_at_zero < 0.0)
                ++always;
            else
                ++never;

        } else if (extra_rounds > 0) {
            double cutoff =
                -delta_at_zero / (double)extra_rounds;

            if (cutoff > 0.0)
                cutoffs[cutoff_count++] = cutoff;
            else
                ++never;

        } else {
            fprintf(
                stderr,
                "FAIL: progressive has fewer rounds than Full8\n"
            );
            exit(EXIT_FAILURE);
        }
    }

    printf(
        "smooth8 sel=50%% tol=%3.1f%% %-3s "
        "BW=%5.2f MB/s | "
        "no-RTT-limit=%d never=%d "
        "positive-cutoffs=%zu",

        tol == 0 ? 1.0 : 0.1,
        joint_policy_names[policy],
        bandwidth,
        always,
        never,
        cutoff_count
    );

    if (cutoff_count != 0) {
        printf(
            " cutoff-median=%.4f ms",
            joint_median(cutoffs, cutoff_count)
        );
    }

    puts("");
}


int main(void)
{
    const char *family_names[FAMILIES] = {
        "temperature-like",
        "wind-like"
    };

    puts("Oracle J: paired CPU + per-round cost model");
    puts("CPU: locally measured decode + COUNT + scheduling.");
    puts("Transfer and RTT: hypothetical, NOT GCP measurements.");
    puts("Only MAYBE-block query stage is included.");
    puts("");

    for (int family = 0; family < FAMILIES; ++family) {
        for (int s = 0; s < SEEDS; ++s) {
            for (int pc = 0; pc < JOINT_PASSES; ++pc) {

                int pass_index = joint_pass_indices[pc];

                if (!prepare_sample(
                        family,
                        seeds[s],
                        passes[pass_index]
                    )) {
                    fprintf(stderr, "FAIL: prepare sample\n");
                    return 1;
                }

                for (int qc = 0; qc < JOINT_QUERIES; ++qc) {

                    int query_index = joint_query_indices[qc];
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
                            fprintf(stderr, "FAIL: reference policies\n");
                            return 1;
                        }

                        for (int policy = 0;
                             policy < JOINT_POLICIES;
                             ++policy) {

                            JointTrace trace;

                            if (!joint_execute(
                                    &q,
                                    divisors[tol],
                                    policy,
                                    &trace
                                ) ||
                                !joint_check(
                                    &q,
                                    divisors[tol],
                                    policy,
                                    &trace,
                                    references
                                )) {

                                fprintf(
                                    stderr,
                                    "FAIL: cross-check "
                                    "family=%d seed=%d pass=%d "
                                    "sel=%.2f tol=%zu policy=%s\n",
                                    family,
                                    s,
                                    passes[pass_index],
                                    selectivities[query_index],
                                    divisors[tol],
                                    joint_policy_names[policy]
                                );

                                return 1;
                            }

                            JointSample *sample =
                                &joint_samples
                                    [family]
                                    [pc]
                                    [qc]
                                    [tol]
                                    [policy]
                                    [s];

                            sample->result = trace.result;
                            sample->waves =
                                joint_waves(&trace, 64u);
                        }

                        for (int slot = 0;
                             slot < JOINT_POLICIES;
                             ++slot) {

                            int policy =
                                (slot + s) % JOINT_POLICIES;

                            JointSample *sample =
                                &joint_samples
                                    [family]
                                    [pc]
                                    [qc]
                                    [tol]
                                    [policy]
                                    [s];

                            if (!joint_benchmark(
                                    &q,
                                    divisors[tol],
                                    policy,
                                    &sample->cpu_us
                                )) {

                                fprintf(
                                    stderr,
                                    "FAIL: CPU timing policy=%s\n",
                                    joint_policy_names[policy]
                                );
                                return 1;
                            }
                        }
                    }
                }
            }
        }
    }

    for (int scenario_index = 0;
         scenario_index < JOINT_SCENARIOS;
         ++scenario_index) {

        const JointScenario *scenario =
            &joint_scenarios[scenario_index];

        printf(
            "\n========== %s ==========\n",
            scenario->name
        );

        printf(
            "round=%.2f ms bandwidth=%.2f MB/s "
            "wave=%.2f ms parallelism=%zu\n",

            scenario->latency_ms,
            scenario->bandwidth_mb_s,
            scenario->wave_ms,
            scenario->parallelism
        );

        for (int family = 0;
             family < FAMILIES;
             ++family) {

            printf("\n--- %s ---\n", family_names[family]);

            for (int pc = 0; pc < JOINT_PASSES; ++pc) {
                for (int qc = 0; qc < JOINT_QUERIES; ++qc) {
                    for (int tol = 0; tol < TOLS; ++tol) {
                        joint_print_case(
                            scenario_index,
                            family,
                            pc,
                            qc,
                            tol
                        );
                    }
                }
            }
        }
    }

    puts("\n========== BREAK-EVEN RTT ==========");
    puts("Only smooth8 / 50% selectivity.");
    puts("A positive cutoff means benefit only BELOW that RTT.");
    puts("no-RTT-limit means equal rounds and lower modeled cost.");
    puts("never means no benefit for any nonnegative RTT.");

    const double bandwidths[] = {0.25, 50.0};

    for (int family = 0; family < FAMILIES; ++family) {
        printf("\n--- %s ---\n", family_names[family]);

        for (int tol = 0; tol < TOLS; ++tol) {
            for (size_t bw = 0;
                 bw < sizeof(bandwidths) / sizeof(bandwidths[0]);
                 ++bw) {

                joint_print_break_even(
                    family,
                    tol,
                    JOINT_B64,
                    bandwidths[bw]
                );

                joint_print_break_even(
                    family,
                    tol,
                    JOINT_P64,
                    bandwidths[bw]
                );
            }
        }
    }

    printf(
        "\nBenchmark checksum: %" PRIu64 "\n",
        (uint64_t)joint_sink
    );

    puts("ALL PAIRED JOINT-COST CHECKS PASSED");
    puts("Model time is NOT measured cloud latency.");

    return 0;
}