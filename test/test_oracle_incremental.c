

#define _POSIX_C_SOURCE 200809L

#define main oracle_batch_embedded_main
#include "test_oracle_batch.c"
#undef main

#include <inttypes.h>
#include <time.h>

enum {
    INC_PASS_CASES = 2,
    INC_QUERY_CASES = 2,
    INC_POLICIES = 2,

    INC_TRIALS = 7,
    INC_REPEATS = 10,

    INC_MAX_EVENTS = LEVELS * BLOCKS,
    INC_MAX_PHASES = 1 + 3 * BLOCKS
};

static const int inc_pass_indices[INC_PASS_CASES] = {0, 4};

static const int inc_query_indices[INC_QUERY_CASES] = {0, 2};

static const char* inc_policy_names[INC_POLICIES] = {"B64", "P64"};

typedef struct {
    size_t block_id;
    size_t level;
    Decision expected;
} IncEvent;

typedef struct {
    size_t end_event;

    size_t lower;
    size_t unknown;

    size_t cumulative_bytes;
    size_t cumulative_reads;
} IncPhase;

typedef struct {
    IncEvent events[INC_MAX_EVENTS];
    IncPhase phases[INC_MAX_PHASES];

    size_t event_count;
    size_t phase_count;

    StrategyResult reference;
} IncTrace;

typedef struct {
    double lower[BLOCK_VALUES];
    double upper[BLOCK_VALUES];

    unsigned char status[BLOCK_VALUES];
    unsigned char unknown_ids[BLOCK_VALUES];

    size_t in;
    size_t unknown;
    size_t level;
} IncBlockCache;

/* Static storage avoids a large stack allocation. */
static IncTrace inc_trace;
static IncBlockCache inc_cache[BLOCKS];

static volatile uint64_t inc_sink = 0u;

static double inc_us[FAMILIES][INC_PASS_CASES][INC_QUERY_CASES][TOLS][INC_POLICIES][2]
                    [SEEDS];

static double inc_speedup[FAMILIES][INC_PASS_CASES][INC_QUERY_CASES][TOLS][INC_POLICIES]
                         [SEEDS];

static double inc_event_counts[FAMILIES][INC_PASS_CASES][INC_QUERY_CASES][TOLS]
                              [INC_POLICIES][SEEDS];

static double inc_round_counts[FAMILIES][INC_PASS_CASES][INC_QUERY_CASES][TOLS]
                              [INC_POLICIES][SEEDS];

static int
inc_add_event(IncTrace* t, size_t block_id, size_t level, Decision expected) {
    if (t->event_count >= INC_MAX_EVENTS) {
        fprintf(stderr, "FAIL: event capacity exceeded\n");
        return 0;
    }

    IncEvent* e = &t->events[t->event_count++];

    e->block_id = block_id;
    e->level = level;
    e->expected = expected;

    return 1;
}

static int inc_add_phase(
    IncTrace* t,
    size_t lower,
    size_t unknown,
    size_t bytes,
    size_t reads,
    const Query* q
) {
    if (t->phase_count >= INC_MAX_PHASES || !valid_interval(lower, unknown, q->truth)) {

        fprintf(stderr, "FAIL: invalid phase\n");
        return 0;
    }

    IncPhase* p = &t->phases[t->phase_count++];

    p->end_event = t->event_count;
    p->lower = lower;
    p->unknown = unknown;
    p->cumulative_bytes = bytes;
    p->cumulative_reads = reads;

    return 1;
}

static int inc_build_trace(const Query* q, size_t divisor, int predicted, IncTrace* t) {
    memset(t, 0, sizeof(*t));

    if (!run_online(q, divisor, 64u, predicted, &t->reference)) {
        fprintf(stderr, "FAIL: original online policy\n");
        return 0;
    }

    size_t level[BLOCKS] = {0};
    Decision current[BLOCKS];

    size_t lower = q->metadata_in;
    size_t unknown = 0u;
    size_t bytes = 0u;
    size_t reads = 0u;

    if (q->maybe == 0u) {
        return t->reference.bytes == 0u && t->reference.reads == 0u &&
               t->reference.rounds == 0u && t->reference.lower == lower &&
               t->reference.unknown == 0u;
    }

    for (size_t j = 0; j < q->maybe; ++j) {
        size_t b = q->ids[j];

        level[j] = predicted ? predict_initial_level(q, b, divisor) : 0u;

        current[j] = decide_block(b, level[j], q->threshold);

        if (!inc_add_event(t, b, level[j], current[j])) {
            return 0;
        }

        lower += current[j].in;
        unknown += current[j].unknown;

        bytes += LEVEL_BYTES * (level[j] + 1u);
        ++reads;
    }

    if (!inc_add_phase(t, lower, unknown, bytes, reads, q)) {
        return 0;
    }

    while (!meets_tolerance(lower, unknown, divisor)) {
        unsigned char selected[BLOCKS] = {0};
        size_t batch[64];
        size_t n = 0u;

        for (size_t slot = 0; slot < 64u; ++slot) {
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
            fprintf(stderr, "FAIL: no refinable block\n");
            return 0;
        }

        /*
         * Each selected block receives ONE additional
         * 16-byte prefix in this phase.
         */
        for (size_t k = 0; k < n; ++k) {
            size_t j = batch[k];
            size_t b = q->ids[j];

            Decision before = current[j];

            ++level[j];

            Decision after = decide_block(b, level[j], q->threshold);

            if (after.in < before.in || after.unknown > before.unknown) {

                fprintf(stderr, "FAIL: original decisions not monotone\n");
                return 0;
            }

            if (!inc_add_event(t, b, level[j], after)) {
                return 0;
            }

            lower += after.in - before.in;
            unknown -= before.unknown - after.unknown;

            current[j] = after;
        }

        bytes += n * LEVEL_BYTES;
        reads += n;

        if (!inc_add_phase(t, lower, unknown, bytes, reads, q)) {
            return 0;
        }
    }

    /*
     * Exact equality with the pre-existing strategy.
     */
    if (bytes != t->reference.bytes || reads != t->reference.reads ||
        t->phase_count != t->reference.rounds || lower != t->reference.lower ||
        unknown != t->reference.unknown || t->event_count != reads) {

        fprintf(stderr, "FAIL: event trace differs from original policy\n");
        return 0;
    }

    return 1;
}

static int
inc_apply_level(IncBlockCache* c, size_t b, size_t level, double threshold, int first) {
    double eps = chunk_eps[level];

    if (first) {
        c->in = 0u;
        c->unknown = 0u;

        for (size_t i = 0; i < BLOCK_VALUES; ++i) {
            double v = (double)decoded[level][b][i];

            double lo = eps == 0.0 ? v : nextafter(v - eps, -INFINITY);

            double hi = eps == 0.0 ? v : nextafter(v + eps, INFINITY);

            c->lower[i] = lo;
            c->upper[i] = hi;

            if (lo > threshold) {
                c->status[i] = 1u;
                ++c->in;

            } else if (hi > threshold) {
                c->status[i] = 0u;

                c->unknown_ids[c->unknown++] = (unsigned char)i;

            } else {
                c->status[i] = 2u;
            }
        }

        c->level = level;
        return 1;
    }

    size_t remaining = 0u;

    /*
     * Compact unknown_ids[] in place.
     * The write position never exceeds read position.
     */
    for (size_t k = 0; k < c->unknown; ++k) {
        size_t i = c->unknown_ids[k];

        double v = (double)decoded[level][b][i];

        double new_lo = eps == 0.0 ? v : nextafter(v - eps, -INFINITY);

        double new_hi = eps == 0.0 ? v : nextafter(v + eps, INFINITY);

        double lo = fmax(c->lower[i], new_lo);
        double hi = fmin(c->upper[i], new_hi);

        if (lo > hi) {
            fprintf(stderr, "FAIL: empty incremental interval\n");
            return 0;
        }

        c->lower[i] = lo;
        c->upper[i] = hi;

        if (lo > threshold) {
            c->status[i] = 1u;
            ++c->in;

        } else if (hi > threshold) {
            c->status[i] = 0u;

            c->unknown_ids[remaining++] = (unsigned char)i;

        } else {
            c->status[i] = 2u;
        }
    }

    c->unknown = remaining;
    c->level = level;

    return 1;
}

static int
inc_apply_event(const IncEvent* e, unsigned char started[BLOCKS], double threshold) {
    size_t b = e->block_id;
    size_t target = e->level;

    if (b >= BLOCKS || target >= LEVELS)
        return 0;

    IncBlockCache* c = &inc_cache[b];

    if (!started[b]) {
        if (!inc_apply_level(c, b, 0u, threshold, 1)) {
            return 0;
        }

        for (size_t level = 1u; level <= target; ++level) {

            if (!inc_apply_level(c, b, level, threshold, 0)) {
                return 0;
            }
        }

        started[b] = 1u;

    } else {
        if (target != c->level + 1u)
            return 0;

        if (!inc_apply_level(c, b, target, threshold, 0)) {
            return 0;
        }
    }

    return 1;
}

static int inc_validate_trace(const IncTrace* t, const Query* q, size_t divisor) {
    unsigned char started[BLOCKS] = {0};
    size_t phase = 0u;

    for (size_t eidx = 0; eidx < t->event_count; ++eidx) {

        const IncEvent* e = &t->events[eidx];

        if (!inc_apply_event(e, started, q->threshold)) {
            fprintf(stderr, "FAIL: incremental event update\n");
            return 0;
        }

        size_t b = e->block_id;
        const IncBlockCache* c = &inc_cache[b];

        if (c->in != e->expected.in || c->unknown != e->expected.unknown) {

            fprintf(
                stderr,
                "FAIL: block decision differs "
                "block=%zu level=%zu "
                "old=%zu/%zu incremental=%zu/%zu\n",

                b,
                e->level,

                e->expected.in,
                e->expected.unknown,

                c->in,
                c->unknown
            );
            return 0;
        }

        size_t verified_in = 0u;
        size_t verified_unknown = 0u;

        for (size_t i = 0; i < BLOCK_VALUES; ++i) {
            int truth = (double)decoded[3][b][i] > q->threshold;

            if (c->status[i] == 1u) {
                if (!truth) {
                    fprintf(stderr, "FAIL: false IN\n");
                    return 0;
                }

                ++verified_in;

            } else if (c->status[i] == 2u) {
                if (truth) {
                    fprintf(stderr, "FAIL: false OUT\n");
                    return 0;
                }

            } else if (c->status[i] == 0u) {
                ++verified_unknown;

            } else {
                fprintf(stderr, "FAIL: invalid status\n");
                return 0;
            }
        }

        if (verified_in != c->in || verified_unknown != c->unknown) {

            fprintf(stderr, "FAIL: incremental cache counts\n");
            return 0;
        }

        if (phase < t->phase_count && eidx + 1u == t->phases[phase].end_event) {

            size_t lower = q->metadata_in;
            size_t unknown = 0u;

            for (size_t j = 0; j < q->maybe; ++j) {
                size_t id = q->ids[j];

                if (!started[id]) {
                    fprintf(stderr, "FAIL: missing initial block\n");
                    return 0;
                }

                lower += inc_cache[id].in;
                unknown += inc_cache[id].unknown;
            }

            const IncPhase* p = &t->phases[phase];

            if (lower != p->lower || unknown != p->unknown ||
                !valid_interval(lower, unknown, q->truth)) {

                fprintf(
                    stderr,
                    "FAIL: global COUNT interval "
                    "differs at phase=%zu\n",
                    phase
                );
                return 0;
            }

            ++phase;
        }
    }

    if (phase != t->phase_count) {
        fprintf(stderr, "FAIL: missing phase validation\n");
        return 0;
    }

    if (t->phase_count != 0u) {
        const IncPhase* last = &t->phases[t->phase_count - 1u];

        if (last->lower != t->reference.lower ||
            last->unknown != t->reference.unknown ||
            last->cumulative_bytes != t->reference.bytes ||
            last->cumulative_reads != t->reference.reads ||
            !meets_tolerance(last->lower, last->unknown, divisor)) {

            fprintf(stderr, "FAIL: final trace result differs\n");
            return 0;
        }
    }

    return 1;
}

static uint64_t inc_clock_ns(void) {
    struct timespec ts;

    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        perror("clock_gettime");
        return 0u;
    }

    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
}

static double inc_median7(const double values[INC_TRIALS]) {
    double copy[INC_TRIALS];

    memcpy(copy, values, sizeof(copy));

    for (int i = 0; i < INC_TRIALS; ++i) {
        for (int j = i + 1; j < INC_TRIALS; ++j) {
            if (copy[j] < copy[i]) {
                double tmp = copy[i];
                copy[i] = copy[j];
                copy[j] = tmp;
            }
        }
    }

    return copy[INC_TRIALS / 2];
}

static int
inc_time_replay(const IncTrace* t, const Query* q, int method, double* us_per_query) {
    if (t->event_count == 0u) {
        *us_per_query = 0.0;
        return 1;
    }

    uint64_t checksum = 0u;
    uint64_t start = inc_clock_ns();

    if (start == 0u)
        return 0;

    for (int rep = 0; rep < INC_REPEATS; ++rep) {
        unsigned char started[BLOCKS] = {0};

        for (size_t eidx = 0; eidx < t->event_count; ++eidx) {

            const IncEvent* e = &t->events[eidx];

            if (method == 0) {
                Decision d = decide_block(e->block_id, e->level, q->threshold);

                checksum += (uint64_t)d.in + (uint64_t)d.unknown;

            } else {
                if (!inc_apply_event(e, started, q->threshold)) {

                    fprintf(stderr, "FAIL: timed incremental replay\n");
                    return 0;
                }

                const IncBlockCache* c = &inc_cache[e->block_id];

                checksum += (uint64_t)c->in + (uint64_t)c->unknown;
            }
        }
    }

    uint64_t end = inc_clock_ns();

    if (end <= start) {
        fprintf(stderr, "FAIL: invalid CPU timer\n");
        return 0;
    }

    /*
     * Publish the checksum after timing.
     * Both methods consume their actual results.
     */
    inc_sink += checksum;

    /*
     * This is measured predicate CPU per query replay.
     * It excludes the trace-building and validation
     * performed outside the timed section.
     */
    *us_per_query = (double)(end - start) / ((double)INC_REPEATS * 1000.0);

    return 1;
}

static int inc_measure_pair(
    const IncTrace* t,
    const Query* q,
    double* old_us,
    double* incremental_us
) {
    double old_trials[INC_TRIALS];
    double new_trials[INC_TRIALS];

    /*
     * Warm-up is separate from the seven timed trials.
     */
    double ignored = 0.0;

    if (!inc_time_replay(t, q, 0, &ignored) || !inc_time_replay(t, q, 1, &ignored)) {
        return 0;
    }

    for (int trial = 0; trial < INC_TRIALS; ++trial) {
        if ((trial & 1) == 0) {
            if (!inc_time_replay(t, q, 0, &old_trials[trial]) ||
                !inc_time_replay(t, q, 1, &new_trials[trial])) {
                return 0;
            }

        } else {
            if (!inc_time_replay(t, q, 1, &new_trials[trial]) ||
                !inc_time_replay(t, q, 0, &old_trials[trial])) {
                return 0;
            }
        }
    }

    *old_us = inc_median7(old_trials);
    *incremental_us = inc_median7(new_trials);

    return *old_us > 0.0 && *incremental_us > 0.0;
}

static void inc_print_summary(int family, int pc, int qc, int tol, int policy) {
    printf(
        "pass=%d sel=%2.0f%% tol=%3.1f%% %-3s | "
        "old=%.2f us "
        "incremental=%.2f us "
        "CPU ratio=%.2fx | "
        "events=%.0f rounds=%.0f\n",

        passes[inc_pass_indices[pc]],
        100.0 * selectivities[inc_query_indices[qc]],
        tol == 0 ? 1.0 : 0.1,
        inc_policy_names[policy],

        median5(inc_us[family][pc][qc][tol][policy][0]),

        median5(inc_us[family][pc][qc][tol][policy][1]),

        median5(inc_speedup[family][pc][qc][tol][policy]),

        median5(inc_event_counts[family][pc][qc][tol][policy]),

        median5(inc_round_counts[family][pc][qc][tol][policy])
    );
}

int main(void) {
    const char* family_names[FAMILIES] = {"temperature-like", "wind-like"};

    puts("Oracle H: incremental certified COUNT");
    puts("OLD vs incremental: SAME B64/P64 event sequences.");
    puts("Every block event and every round is validated.");
    puts("Timing = predicate CPU only, NOT end-to-end time.");
    puts("Ground truth = fully decoded STORED 8-bit data.");
    puts("No GCP requests or existing-file modifications.");
    puts("");

    for (int family = 0; family < FAMILIES; ++family) {
        for (int seed_index = 0; seed_index < SEEDS; ++seed_index) {

            for (int pc = 0; pc < INC_PASS_CASES; ++pc) {

                int pass_index = inc_pass_indices[pc];

                if (!prepare_sample(family, seeds[seed_index], passes[pass_index])) {

                    fprintf(
                        stderr,
                        "FAIL: sample family=%d seed=%d "
                        "passes=%d\n",
                        family,
                        seed_index,
                        passes[pass_index]
                    );
                    return 1;
                }

                for (int qc = 0; qc < INC_QUERY_CASES; ++qc) {

                    int query_index = inc_query_indices[qc];

                    Query q;

                    if (!build_query(selectivities[query_index], &q)) {

                        fprintf(stderr, "FAIL: build query\n");
                        return 1;
                    }

                    for (int tol = 0; tol < TOLS; ++tol) {
                        for (int policy = 0; policy < INC_POLICIES; ++policy) {

                            int predicted = policy == 1;

                            if (!inc_build_trace(
                                    &q, divisors[tol], predicted, &inc_trace
                                )) {

                                fprintf(
                                    stderr,
                                    "FAIL: build trace "
                                    "family=%d seed=%d "
                                    "passes=%d sel=%.2f "
                                    "tol=%zu policy=%s\n",

                                    family,
                                    seed_index,
                                    passes[pass_index],
                                    selectivities[query_index],
                                    divisors[tol],
                                    inc_policy_names[policy]
                                );
                                return 1;
                            }

                            if (!inc_validate_trace(&inc_trace, &q, divisors[tol])) {

                                fprintf(
                                    stderr,
                                    "FAIL: validation "
                                    "family=%d seed=%d "
                                    "passes=%d sel=%.2f "
                                    "tol=%zu policy=%s\n",

                                    family,
                                    seed_index,
                                    passes[pass_index],
                                    selectivities[query_index],
                                    divisors[tol],
                                    inc_policy_names[policy]
                                );
                                return 1;
                            }

                            double old_us;
                            double new_us;

                            if (!inc_measure_pair(&inc_trace, &q, &old_us, &new_us)) {

                                fprintf(
                                    stderr,
                                    "FAIL: timing policy=%s\n",
                                    inc_policy_names[policy]
                                );
                                return 1;
                            }

                            inc_us[family][pc][qc][tol][policy][0][seed_index] = old_us;

                            inc_us[family][pc][qc][tol][policy][1][seed_index] = new_us;

                            inc_speedup[family][pc][qc][tol][policy][seed_index] =
                                old_us / new_us;

                            inc_event_counts[family][pc][qc][tol][policy][seed_index] =
                                (double)inc_trace.event_count;

                            inc_round_counts[family][pc][qc][tol][policy][seed_index] =
                                (double)inc_trace.phase_count;
                        }
                    }
                }
            }
        }
    }

    for (int family = 0; family < FAMILIES; ++family) {
        printf("\n========== %s ==========\n", family_names[family]);

        for (int pc = 0; pc < INC_PASS_CASES; ++pc) {
            for (int qc = 0; qc < INC_QUERY_CASES; ++qc) {
                for (int tol = 0; tol < TOLS; ++tol) {
                    for (int policy = 0; policy < INC_POLICIES; ++policy) {

                        inc_print_summary(family, pc, qc, tol, policy);
                    }
                }
            }
        }
    }

    printf("\nBenchmark checksum: %" PRIu64 "\n", (uint64_t)inc_sink);

    puts("ALL INCREMENTAL COUNT CHECKS PASSED");
    puts("Reminder: CPU ratio covers predicate replay only; "
         "it is NOT overall query speedup.");

    return 0;
}