

#define main oracle_batch_embedded_main
#include "test_oracle_batch.c"
#undef main

#include <float.h>

enum {
    COST_SCENARIOS = 4,
    COST_PASS_CASES = 2,
    COST_QUERY_CASES = 2,
    COST_POLICIES = POLICIES + 1,
    COST_FULL8 = POLICIES,

    TRACE_MAX_ROUNDS = 1 + 3 * BLOCKS
};

static const int cost_pass_indices[COST_PASS_CASES] = {0, 4};

static const int cost_query_indices[COST_QUERY_CASES] = {0, 2};

typedef struct {
    const char* name;
    double round_latency_ms;
    double bandwidth_mb_per_s;
    double wave_service_ms;
    size_t parallelism;
} CostScenario;

static const CostScenario scenarios[COST_SCENARIOS] = {
    {"LOW_RTT_SLOW_LINK", 0.05, 0.25, 0.02, 64u},
    {"LOW_RTT_FAST_LINK", 0.05, 50.0, 0.02, 64u},
    {"MID_RTT_FAST_LINK", 5.0, 50.0, 0.02, 64u},
    {"HIGH_RTT_FAST_LINK", 50.0, 50.0, 0.02, 64u}
};

static const char* cost_policy_names[COST_POLICIES] =
    {"U", "B16", "B32", "B64", "B128", "P64", "Full8"};

typedef struct {
    StrategyResult total;

    size_t count;
    size_t round_bytes[TRACE_MAX_ROUNDS];
    size_t round_reads[TRACE_MAX_ROUNDS];
} RoundTrace;

static double modeled_ms[COST_SCENARIOS][FAMILIES][COST_PASS_CASES][COST_QUERY_CASES]
                        [TOLS][COST_POLICIES][SEEDS];

static double modeled_waves[FAMILIES][COST_PASS_CASES][COST_QUERY_CASES][TOLS]
                           [COST_POLICIES][SEEDS];

static int record_round(RoundTrace* trace, size_t bytes, size_t reads) {
    if (trace->count >= TRACE_MAX_ROUNDS || bytes == 0u || reads == 0u) {

        fprintf(stderr, "FAIL: invalid round trace\n");
        return 0;
    }

    size_t k = trace->count++;

    trace->round_bytes[k] = bytes;
    trace->round_reads[k] = reads;

    trace->total.bytes += bytes;
    trace->total.reads += reads;
    ++trace->total.rounds;

    return 1;
}

static int trace_full8(const Query* q, RoundTrace* trace) {
    memset(trace, 0, sizeof(*trace));

    if (q->maybe != 0u) {
        if (!record_round(trace, q->static_bytes, q->maybe)) {
            return 0;
        }
    }

    trace->total.lower = q->truth;
    trace->total.unknown = 0u;

    return valid_interval(trace->total.lower, trace->total.unknown, q->truth);
}

static int trace_uniform(const Query* q, size_t divisor, RoundTrace* trace) {
    memset(trace, 0, sizeof(*trace));

    if (q->maybe == 0u) {
        trace->total.lower = q->metadata_in;

        return valid_interval(trace->total.lower, 0u, q->truth);
    }

    for (size_t level = 0; level < LEVELS; ++level) {

        size_t lower = q->metadata_in;
        size_t unknown = 0u;

        for (size_t j = 0; j < q->maybe; ++j) {

            Decision d = decide_block(q->ids[j], level, q->threshold);

            lower += d.in;
            unknown += d.unknown;
        }

        if (!valid_interval(lower, unknown, q->truth)) {

            fprintf(stderr, "FAIL: uniform trace interval\n");
            return 0;
        }

        if (!record_round(trace, q->maybe * LEVEL_BYTES, q->maybe)) {
            return 0;
        }

        if (meets_tolerance(lower, unknown, divisor)) {

            trace->total.lower = lower;
            trace->total.unknown = unknown;
            return 1;
        }
    }

    fprintf(stderr, "FAIL: uniform trace did not converge\n");
    return 0;
}

static int trace_online(
    const Query* q,
    size_t divisor,
    size_t batch_size,
    int pred,
    RoundTrace* trace
) {
    size_t level[BLOCKS] = {0};
    Decision current[BLOCKS];

    size_t lower = q->metadata_in;
    size_t unknown = 0u;

    memset(trace, 0, sizeof(*trace));

    if (batch_size == 0u || batch_size > MAX_BATCH) {

        fprintf(stderr, "FAIL: unsupported batch size\n");
        return 0;
    }

    if (q->maybe == 0u) {
        trace->total.lower = lower;

        return valid_interval(lower, 0u, q->truth);
    }

    size_t initial_bytes = 0u;

    for (size_t j = 0; j < q->maybe; ++j) {

        size_t b = q->ids[j];

        level[j] = pred ? predict_initial_level(q, b, divisor) : 0u;

        current[j] = decide_block(b, level[j], q->threshold);

        lower += current[j].in;
        unknown += current[j].unknown;

        initial_bytes += LEVEL_BYTES * (level[j] + 1u);
    }

    if (!valid_interval(lower, unknown, q->truth)) {

        fprintf(stderr, "FAIL: initial online trace interval\n");
        return 0;
    }

    if (!record_round(trace, initial_bytes, q->maybe)) {
        return 0;
    }

    while (!meets_tolerance(lower, unknown, divisor)) {
        unsigned char selected[BLOCKS] = {0};

        size_t batch[MAX_BATCH];
        size_t n = 0u;

        for (size_t slot = 0; slot < batch_size; ++slot) {

            size_t best = q->maybe;

            for (size_t j = 0; j < q->maybe; ++j) {

                if (selected[j] || level[j] + 1u >= LEVELS ||
                    current[j].unknown == 0u) {
                    continue;
                }

                if (best == q->maybe || current[j].unknown > current[best].unknown ||
                    (current[j].unknown == current[best].unknown &&
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
            fprintf(stderr, "FAIL: online trace cannot refine\n");
            return 0;
        }

        for (size_t k = 0; k < n; ++k) {

            size_t j = batch[k];
            size_t b = q->ids[j];

            Decision before = current[j];

            ++level[j];

            Decision after = decide_block(b, level[j], q->threshold);

            if (after.in < before.in || after.unknown > before.unknown) {

                fprintf(stderr, "FAIL: nonmonotone online trace\n");
                return 0;
            }

            lower += after.in - before.in;

            unknown -= before.unknown - after.unknown;

            current[j] = after;
        }

        if (!valid_interval(lower, unknown, q->truth)) {

            fprintf(stderr, "FAIL: online trace interval\n");
            return 0;
        }

        if (!record_round(trace, n * LEVEL_BYTES, n)) {
            return 0;
        }
    }

    trace->total.lower = lower;
    trace->total.unknown = unknown;

    if (trace->total.bytes > q->static_bytes) {
        fprintf(stderr, "FAIL: trace exceeds full payload\n");
        return 0;
    }

    return 1;
}

static int check_trace(
    const RoundTrace* trace,
    const StrategyResult* reference,
    size_t truth,
    size_t divisor,
    const char* name
) {
    size_t sum_bytes = 0u;
    size_t sum_reads = 0u;

    for (size_t k = 0; k < trace->count; ++k) {

        sum_bytes += trace->round_bytes[k];
        sum_reads += trace->round_reads[k];
    }

    if (sum_bytes != trace->total.bytes || sum_reads != trace->total.reads ||
        trace->count != trace->total.rounds || trace->total.bytes != reference->bytes ||
        trace->total.reads != reference->reads ||
        trace->total.rounds != reference->rounds ||
        trace->total.lower != reference->lower ||
        trace->total.unknown != reference->unknown ||
        !valid_interval(trace->total.lower, trace->total.unknown, truth) ||
        !meets_tolerance(trace->total.lower, trace->total.unknown, divisor)) {

        fprintf(stderr, "FAIL: round trace differs from %s\n", name);

        fprintf(
            stderr,
            "trace: bytes=%zu reads=%zu rounds=%zu "
            "lower=%zu unknown=%zu\n",
            trace->total.bytes,
            trace->total.reads,
            trace->total.rounds,
            trace->total.lower,
            trace->total.unknown
        );

        fprintf(
            stderr,
            "reference: bytes=%zu reads=%zu rounds=%zu "
            "lower=%zu unknown=%zu\n",
            reference->bytes,
            reference->reads,
            reference->rounds,
            reference->lower,
            reference->unknown
        );

        return 0;
    }

    return 1;
}

static size_t round_waves(size_t reads, size_t parallelism) {
    if (reads == 0u)
        return 0u;

    return reads / parallelism + (reads % parallelism != 0u);
}

static size_t total_waves(const RoundTrace* trace, size_t parallelism) {
    size_t total = 0u;

    for (size_t k = 0; k < trace->count; ++k) {

        total += round_waves(trace->round_reads[k], parallelism);
    }

    return total;
}

static double round_modeled_ms(const RoundTrace* trace, const CostScenario* scenario) {
    if (!(scenario->bandwidth_mb_per_s > 0.0) || scenario->parallelism == 0u) {
        return INFINITY;
    }

    double total_ms = 0.0;

    for (size_t k = 0; k < trace->count; ++k) {

        size_t waves = round_waves(trace->round_reads[k], scenario->parallelism);

        double transfer_ms = 1000.0 * (double)trace->round_bytes[k] /
                             (scenario->bandwidth_mb_per_s * 1000000.0);

        total_ms += scenario->round_latency_ms +
                    (double)waves * scenario->wave_service_ms + transfer_ms;
    }

    return total_ms;
}

static void
print_case(int scenario_index, int family, int pass_case, int query_case, int tol) {
    double median[COST_POLICIES];

    int minimum_policy = COST_FULL8;
    double minimum_value = DBL_MAX;

    for (int policy = 0; policy < COST_POLICIES; ++policy) {

        median[policy] = median5(
            modeled_ms[scenario_index][family][pass_case][query_case][tol][policy]
        );

        if (median[policy] < minimum_value - 1e-9) {

            minimum_value = median[policy];
            minimum_policy = policy;

        } else if (fabs(median[policy] - minimum_value) <= 1e-9 &&
                   policy == COST_FULL8) {

            minimum_policy = COST_FULL8;
        }
    }

    double full8_waves =
        median5(modeled_waves[family][pass_case][query_case][tol][COST_FULL8]);

    double b64_waves =
        median5(modeled_waves[family][pass_case][query_case][tol][POLICY_B64]);

    double p64_waves =
        median5(modeled_waves[family][pass_case][query_case][tol][POLICY_P64]);

    printf(
        "pass=%d sel=%2.0f%% tol=%3.1f%% | "
        "Full8=%.3f "
        "U=%.3f "
        "B16=%.3f "
        "B32=%.3f "
        "B64=%.3f "
        "B128=%.3f "
        "P64=%.3f | "
        "waves F8/B64/P64=%.0f/%.0f/%.0f | "
        "MODEL_MIN=%s\n",

        passes[cost_pass_indices[pass_case]],
        100.0 * selectivities[cost_query_indices[query_case]],
        tol == 0 ? 1.0 : 0.1,

        median[COST_FULL8],
        median[POLICY_UNIFORM],
        median[POLICY_B16],
        median[POLICY_B32],
        median[POLICY_B64],
        median[POLICY_B128],
        median[POLICY_P64],

        full8_waves,
        b64_waves,
        p64_waves,

        cost_policy_names[minimum_policy]
    );
}

int main(void) {
    const char* family_names[FAMILIES] = {"temperature-like", "wind-like"};

    const size_t batches[4] = {16u, 32u, 64u, 128u};

    puts("Oracle F: per-round cost sensitivity");
    puts("MODELED milliseconds only; no cloud measurements.");
    puts("Reference = fully decoded STORED 8-bit values.");
    puts("All six strategies are cross-checked against Oracle D.");
    puts("Waves are summed separately across dependent rounds.");
    puts("Reads are uncoalesced logical block-range reads.");
    puts("");

    for (int family = 0; family < FAMILIES; ++family) {

        for (int s = 0; s < SEEDS; ++s) {

            for (int pc = 0; pc < COST_PASS_CASES; ++pc) {

                int pass_index = cost_pass_indices[pc];

                if (!prepare_sample(family, seeds[s], passes[pass_index])) {

                    fprintf(
                        stderr,
                        "FAIL: prepare family=%d seed=%d pass=%d\n",
                        family,
                        s,
                        passes[pass_index]
                    );
                    return 1;
                }

                for (int qc = 0; qc < COST_QUERY_CASES; ++qc) {

                    int query_index = cost_query_indices[qc];

                    Query q;

                    if (!build_query(selectivities[query_index], &q)) {

                        fprintf(stderr, "FAIL: build query\n");
                        return 1;
                    }

                    for (int tol = 0; tol < TOLS; ++tol) {

                        size_t divisor = divisors[tol];

                        StrategyResult reference[POLICIES];

                        if (!evaluate_policies(&q, divisor, reference)) {

                            fprintf(stderr, "FAIL: original policies\n");
                            return 1;
                        }

                        for (int policy = 0; policy < COST_POLICIES; ++policy) {

                            RoundTrace trace;
                            int ok = 0;

                            if (policy == POLICY_UNIFORM) {
                                ok = trace_uniform(&q, divisor, &trace);

                            } else if (policy >= POLICY_B16 && policy <= POLICY_B128) {
                                ok = trace_online(
                                    &q, divisor, batches[policy - POLICY_B16], 0, &trace
                                );

                            } else if (policy == POLICY_P64) {
                                ok = trace_online(&q, divisor, 64u, 1, &trace);

                            } else if (policy == COST_FULL8) {
                                ok = trace_full8(&q, &trace);
                            }

                            if (!ok) {
                                fprintf(
                                    stderr,
                                    "FAIL: trace policy=%s "
                                    "family=%d seed=%d "
                                    "pass=%d sel=%.2f tol=%zu\n",
                                    cost_policy_names[policy],
                                    family,
                                    s,
                                    passes[pass_index],
                                    selectivities[query_index],
                                    divisor
                                );
                                return 1;
                            }

                            if (policy < POLICIES) {
                                if (!check_trace(
                                        &trace,
                                        &reference[policy],
                                        q.truth,
                                        divisor,
                                        cost_policy_names[policy]
                                    )) {
                                    return 1;
                                }

                            } else {

                                if (trace.total.bytes != q.static_bytes ||
                                    trace.total.reads != q.maybe ||
                                    trace.total.rounds != (q.maybe == 0u ? 0u : 1u) ||
                                    trace.total.unknown != 0u ||
                                    !valid_interval(
                                        trace.total.lower, trace.total.unknown, q.truth
                                    )) {
                                    fprintf(stderr, "FAIL: Full8 trace\n");
                                    return 1;
                                }
                            }

                            for (int scenario_index = 0;
                                 scenario_index < COST_SCENARIOS;
                                 ++scenario_index) {

                                double ms = round_modeled_ms(
                                    &trace, &scenarios[scenario_index]
                                );

                                if (!isfinite(ms)) {
                                    fprintf(stderr, "FAIL: non-finite model time\n");
                                    return 1;
                                }

                                modeled_ms[scenario_index][family][pc][qc][tol][policy]
                                          [s] = ms;
                            }

                            modeled_waves[family][pc][qc][tol][policy][s] =
                                (double)total_waves(&trace, scenarios[0].parallelism);
                        }
                    }
                }
            }
        }
    }

    for (int scenario_index = 0; scenario_index < COST_SCENARIOS; ++scenario_index) {

        const CostScenario* scenario = &scenarios[scenario_index];

        printf("\n========== %s ==========\n", scenario->name);

        printf(
            "assumptions: round=%.2f ms, "
            "bandwidth=%.2f MB/s, "
            "wave_service=%.2f ms, "
            "parallelism=%zu\n",

            scenario->round_latency_ms,
            scenario->bandwidth_mb_per_s,
            scenario->wave_service_ms,
            scenario->parallelism
        );

        for (int family = 0; family < FAMILIES; ++family) {

            printf("\n--- %s ---\n", family_names[family]);

            for (int pc = 0; pc < COST_PASS_CASES; ++pc) {

                for (int qc = 0; qc < COST_QUERY_CASES; ++qc) {

                    for (int tol = 0; tol < TOLS; ++tol) {

                        print_case(scenario_index, family, pc, qc, tol);
                    }
                }
            }
        }
    }

    puts("\nALL PER-ROUND TRACE CHECKS PASSED");
    puts("Reminder: modeled time is NOT measured GCP latency.");

    return 0;
}