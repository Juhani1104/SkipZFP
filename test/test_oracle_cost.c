
#define main oracle_batch_embedded_main
#include "test_oracle_batch.c"
#undef main

#include <float.h>

enum {
    COST_SCENARIOS = 4,
    COST_PASS_CASES = 2,
    COST_QUERY_CASES = 2,
    COST_POLICIES = POLICIES + 1,
    FULL8_INDEX = POLICIES
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

static double modeled_ms[COST_SCENARIOS][FAMILIES][COST_PASS_CASES][COST_QUERY_CASES]
                        [TOLS][COST_POLICIES][SEEDS];

static double modeled_time_ms(const StrategyResult* r, const CostScenario* s) {
    if (s->bandwidth_mb_per_s <= 0.0 || s->parallelism == 0u) {
        return INFINITY;
    }

    size_t waves = 0u;

    if (r->reads != 0u) {
        waves = r->reads / s->parallelism + (r->reads % s->parallelism != 0u);
    }

    double transfer_ms =
        1000.0 * (double)r->bytes / (s->bandwidth_mb_per_s * 1000000.0);

    return (double)r->rounds * s->round_latency_ms +
           (double)waves * s->wave_service_ms + transfer_ms;
}

static StrategyResult make_full8(const Query* q) {
    StrategyResult full;
    memset(&full, 0, sizeof(full));

    full.bytes = q->static_bytes;
    full.reads = q->maybe;
    full.rounds = q->maybe == 0u ? 0u : 1u;

    full.lower = q->truth;
    full.unknown = 0u;

    return full;
}

static double cost_median5(const double values[SEEDS]) {
    double copy[SEEDS];

    memcpy(copy, values, sizeof(copy));

    for (int i = 0; i < SEEDS; ++i) {
        for (int j = i + 1; j < SEEDS; ++j) {
            if (copy[j] < copy[i]) {
                double temp = copy[i];
                copy[i] = copy[j];
                copy[j] = temp;
            }
        }
    }

    return copy[SEEDS / 2];
}

static void print_cost_case(
    int scenario_index,
    int family,
    int pass_case,
    int query_case,
    int tol
) {
    const CostScenario* scenario = &scenarios[scenario_index];

    double medians[COST_POLICIES];

    int minimum_index = FULL8_INDEX;
    double minimum_ms = DBL_MAX;

    for (int policy = 0; policy < COST_POLICIES; ++policy) {

        medians[policy] = cost_median5(
            modeled_ms[scenario_index][family][pass_case][query_case][tol][policy]
        );

        if (medians[policy] < minimum_ms - 1e-9) {
            minimum_ms = medians[policy];
            minimum_index = policy;
        } else if (fabs(medians[policy] - minimum_ms) <= 1e-9 &&
                   policy == FULL8_INDEX) {
            minimum_index = FULL8_INDEX;
        }
    }

    printf(
        "pass=%d sel=%2.0f%% tol=%3.1f%% | "
        "Full8=%.3f "
        "U=%.3f "
        "B16=%.3f "
        "B32=%.3f "
        "B64=%.3f "
        "B128=%.3f "
        "P64=%.3f | "
        "MODEL_MIN=%s\n",

        passes[cost_pass_indices[pass_case]],
        100.0 * selectivities[cost_query_indices[query_case]],
        tol == 0 ? 1.0 : 0.1,

        medians[FULL8_INDEX],
        medians[POLICY_UNIFORM],
        medians[POLICY_B16],
        medians[POLICY_B32],
        medians[POLICY_B64],
        medians[POLICY_B128],
        medians[POLICY_P64],

        cost_policy_names[minimum_index]
    );
}

int main(void) {
    const char* family_names[FAMILIES] = {"temperature-like", "wind-like"};

    puts("Oracle E: hypothetical cost sensitivity");
    puts("Every value below is MODELED ms, not measured ms.");
    puts("Full8 = one-shot 64 bytes per MAYBE block.");
    puts("No real GCP requests are made.");
    puts("");

    for (int family = 0; family < FAMILIES; ++family) {
        for (int seed_index = 0; seed_index < SEEDS; ++seed_index) {

            for (int pass_case = 0; pass_case < COST_PASS_CASES; ++pass_case) {

                int original_pass_index = cost_pass_indices[pass_case];

                if (!prepare_sample(
                        family, seeds[seed_index], passes[original_pass_index]
                    )) {
                    fprintf(
                        stderr,
                        "FAIL: prepare family=%d seed=%d pass=%d\n",
                        family,
                        seed_index,
                        passes[original_pass_index]
                    );
                    return 1;
                }

                for (int query_case = 0; query_case < COST_QUERY_CASES; ++query_case) {

                    int original_query_index = cost_query_indices[query_case];

                    Query q;

                    if (!build_query(selectivities[original_query_index], &q)) {
                        fprintf(stderr, "FAIL: build query\n");
                        return 1;
                    }

                    StrategyResult full = make_full8(&q);

                    if (!valid_interval(full.lower, full.unknown, q.truth)) {
                        fprintf(stderr, "FAIL: Full8 COUNT interval\n");
                        return 1;
                    }

                    for (int tol = 0; tol < TOLS; ++tol) {
                        StrategyResult policies[POLICIES];

                        if (!evaluate_policies(&q, divisors[tol], policies)) {
                            fprintf(
                                stderr,
                                "FAIL: policy family=%d "
                                "seed=%d pass=%d query=%d tol=%d\n",
                                family,
                                seed_index,
                                passes[original_pass_index],
                                original_query_index,
                                tol
                            );
                            return 1;
                        }

                        for (int scenario_index = 0; scenario_index < COST_SCENARIOS;
                             ++scenario_index) {

                            const CostScenario* scenario = &scenarios[scenario_index];

                            for (int policy = 0; policy < POLICIES; ++policy) {

                                modeled_ms[scenario_index][family][pass_case]
                                          [query_case][tol][policy][seed_index] =
                                              modeled_time_ms(
                                                  &policies[policy], scenario
                                              );
                            }

                            modeled_ms[scenario_index][family][pass_case][query_case]
                                      [tol][FULL8_INDEX][seed_index] =
                                          modeled_time_ms(&full, scenario);
                        }
                    }
                }
            }
        }
    }

    for (int scenario_index = 0; scenario_index < COST_SCENARIOS; ++scenario_index) {

        const CostScenario* s = &scenarios[scenario_index];

        printf("\n========== %s ==========\n", s->name);

        printf(
            "assumptions: round=%.2f ms, "
            "bandwidth=%.2f MB/s, "
            "wave_service=%.2f ms, "
            "parallelism=%zu\n",
            s->round_latency_ms,
            s->bandwidth_mb_per_s,
            s->wave_service_ms,
            s->parallelism
        );

        for (int family = 0; family < FAMILIES; ++family) {
            printf("\n--- %s ---\n", family_names[family]);

            for (int pass_case = 0; pass_case < COST_PASS_CASES; ++pass_case) {

                for (int query_case = 0; query_case < COST_QUERY_CASES; ++query_case) {

                    for (int tol = 0; tol < TOLS; ++tol) {
                        print_cost_case(
                            scenario_index, family, pass_case, query_case, tol
                        );
                    }
                }
            }
        }
    }

    puts("\nALL COST MODEL INPUT CHECKS PASSED");
    puts("Reminder: modeled time is NOT cloud latency.");
    return 0;
}