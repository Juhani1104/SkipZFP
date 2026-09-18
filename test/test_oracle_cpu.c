
#define _POSIX_C_SOURCE 200809L

#define main oracle_batch_embedded_main
#include "test_oracle_batch.c"
#undef main

#include <inttypes.h>
#include <time.h>

enum { CPU_PASS_CASES = 2, CPU_QUERY_CASES = 2, CPU_TRIALS = 7, CPU_REPEATS = 6 };

static const int cpu_pass_indices[CPU_PASS_CASES] = {0, 4};

static const int cpu_query_indices[CPU_QUERY_CASES] = {0, 2};

typedef struct {
    double prefix_ns[LEVELS];
    double certified_ns[LEVELS];

    double production_full8_ns;
    double full8_scan_ns;
} CpuMeasure;

typedef struct {
    StrategyResult total;

    size_t decoder_calls[LEVELS];

    size_t decision_calls[LEVELS];
} WorkTrace;

static CpuMeasure measurements[FAMILIES][CPU_PASS_CASES][CPU_QUERY_CASES][SEEDS];

static double query_cpu_us[FAMILIES][CPU_PASS_CASES][CPU_QUERY_CASES][TOLS][4][SEEDS];

static volatile uint64_t benchmark_sink = 0u;

static uint64_t clock_ns(void) {
    struct timespec ts;

    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        perror("clock_gettime");
        return 0u;
    }

    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
}

static double trial_median(const double values[CPU_TRIALS]) {
    double copy[CPU_TRIALS];

    memcpy(copy, values, sizeof(copy));

    for (int i = 0; i < CPU_TRIALS; ++i) {
        for (int j = i + 1; j < CPU_TRIALS; ++j) {
            if (copy[j] < copy[i]) {
                double temp = copy[i];
                copy[i] = copy[j];
                copy[j] = temp;
            }
        }
    }

    return copy[CPU_TRIALS / 2];
}

static int
decode_once(size_t b, size_t level, int production, float out[BLOCK_VALUES]) {
    const unsigned char* source = packed + b * FULL_BYTES;

    if (production) {
        return szfp_decompress_block(
                   source, FULL_BYTES, 8.0, (int)zfp_type_float, 3, out
               ) == SZ_OK;
    }

    return decode_prefix(source, LEVEL_BYTES * (level + 1u), rates[level], out);
}

static int
measure_decoder(const Query* q, size_t level, int production, double* ns_per_block) {
    float scratch[BLOCK_VALUES];
    double trials[CPU_TRIALS];

    if (q->maybe == 0u) {
        *ns_per_block = 0.0;
        return 1;
    }

    size_t expected_level = production ? 3u : level;

    for (int warmup = 0; warmup < 2; ++warmup) {
        for (size_t j = 0; j < q->maybe; ++j) {
            size_t b = q->ids[j];

            if (!decode_once(b, level, production, scratch)) {

                fprintf(
                    stderr,
                    "FAIL: decoder block=%zu "
                    "level=%zu production=%d\n",
                    b,
                    level,
                    production
                );
                return 0;
            }

            if (memcmp(scratch, decoded[expected_level][b], sizeof(scratch)) != 0) {

                fprintf(
                    stderr,
                    "FAIL: decoder differs from "
                    "prepared reconstruction "
                    "block=%zu level=%zu production=%d\n",
                    b,
                    level,
                    production
                );
                return 0;
            }
        }
    }

    for (int trial = 0; trial < CPU_TRIALS; ++trial) {
        uint64_t checksum = 0u;
        uint64_t start = clock_ns();

        if (start == 0u)
            return 0;

        for (int rep = 0; rep < CPU_REPEATS; ++rep) {
            for (size_t j = 0; j < q->maybe; ++j) {
                size_t b = q->ids[j];

                if (!decode_once(b, level, production, scratch)) {

                    fprintf(stderr, "FAIL: timed decoder\n");
                    return 0;
                }

                uint32_t bits = 0u;
                size_t index = (j + (size_t)rep) % BLOCK_VALUES;

                memcpy(&bits, &scratch[index], sizeof(bits));

                checksum += (uint64_t)bits;
            }
        }

        uint64_t end = clock_ns();

        if (end <= start) {
            fprintf(stderr, "FAIL: invalid timer result\n");
            return 0;
        }

        benchmark_sink += checksum;

        trials[trial] =
            (double)(end - start) / ((double)CPU_REPEATS * (double)q->maybe);
    }

    *ns_per_block = trial_median(trials);
    return 1;
}

static size_t scan_full8_block(size_t b, double threshold) {
    size_t count = 0u;

    for (size_t i = 0; i < BLOCK_VALUES; ++i) {
        if ((double)decoded[3][b][i] > threshold)
            ++count;
    }

    return count;
}

static int
measure_predicate(const Query* q, size_t level, int full8, double* ns_per_block) {
    double trials[CPU_TRIALS];

    if (q->maybe == 0u) {
        *ns_per_block = 0.0;
        return 1;
    }

    size_t lower = q->metadata_in;
    size_t unknown = 0u;

    for (size_t j = 0; j < q->maybe; ++j) {
        size_t b = q->ids[j];

        if (full8) {
            lower += scan_full8_block(b, q->threshold);
        } else {
            Decision d = decide_block(b, level, q->threshold);

            lower += d.in;
            unknown += d.unknown;
        }
    }

    if (!valid_interval(lower, unknown, q->truth)) {

        fprintf(
            stderr,
            "FAIL: invalid predicate interval "
            "level=%zu full8=%d\n",
            level,
            full8
        );
        return 0;
    }

    if (full8 && (lower != q->truth || unknown != 0u)) {
        fprintf(stderr, "FAIL: full8 scan COUNT differs\n");
        return 0;
    }

    for (int warmup = 0; warmup < 2; ++warmup) {
        for (size_t j = 0; j < q->maybe; ++j) {
            size_t b = q->ids[j];

            if (full8) {
                benchmark_sink += scan_full8_block(b, q->threshold);
            } else {
                Decision d = decide_block(b, level, q->threshold);

                benchmark_sink += (uint64_t)d.in + (uint64_t)d.unknown;
            }
        }
    }

    for (int trial = 0; trial < CPU_TRIALS; ++trial) {
        uint64_t checksum = 0u;
        uint64_t start = clock_ns();

        if (start == 0u)
            return 0;

        for (int rep = 0; rep < CPU_REPEATS; ++rep) {
            for (size_t j = 0; j < q->maybe; ++j) {
                size_t b = q->ids[j];

                if (full8) {
                    checksum += scan_full8_block(b, q->threshold);
                } else {
                    Decision d = decide_block(b, level, q->threshold);

                    checksum += (uint64_t)d.in + (uint64_t)d.unknown;
                }
            }
        }

        uint64_t end = clock_ns();

        if (end <= start) {
            fprintf(stderr, "FAIL: invalid timer result\n");
            return 0;
        }

        benchmark_sink += checksum;

        trials[trial] =
            (double)(end - start) / ((double)CPU_REPEATS * (double)q->maybe);
    }

    *ns_per_block = trial_median(trials);
    return 1;
}

static int measure_query_primitives(const Query* q, CpuMeasure* m) {
    memset(m, 0, sizeof(*m));

    for (size_t level = 0; level < LEVELS; ++level) {
        if (!measure_decoder(q, level, 0, &m->prefix_ns[level])) {
            return 0;
        }

        if (!measure_predicate(q, level, 0, &m->certified_ns[level])) {
            return 0;
        }
    }

    if (!measure_decoder(q, 3u, 1, &m->production_full8_ns)) {
        return 0;
    }

    if (!measure_predicate(q, 3u, 1, &m->full8_scan_ns)) {
        return 0;
    }

    return 1;
}

static int work_uniform(const Query* q, size_t divisor, WorkTrace* work) {
    memset(work, 0, sizeof(*work));

    if (q->maybe == 0u) {
        work->total.lower = q->metadata_in;
        return valid_interval(work->total.lower, 0u, q->truth);
    }

    for (size_t level = 0; level < LEVELS; ++level) {
        size_t lower = q->metadata_in;
        size_t unknown = 0u;

        for (size_t j = 0; j < q->maybe; ++j) {
            Decision d = decide_block(q->ids[j], level, q->threshold);

            lower += d.in;
            unknown += d.unknown;
        }

        ++work->total.rounds;
        work->total.bytes += q->maybe * LEVEL_BYTES;
        work->total.reads += q->maybe;

        work->decoder_calls[level] += q->maybe;
        work->decision_calls[level] += q->maybe;

        if (!valid_interval(lower, unknown, q->truth))
            return 0;

        if (meets_tolerance(lower, unknown, divisor)) {
            work->total.lower = lower;
            work->total.unknown = unknown;
            return 1;
        }
    }

    return 0;
}

static int work_online(const Query* q, size_t divisor, int pred, WorkTrace* work) {
    size_t current_level[BLOCKS] = {0};
    Decision current[BLOCKS];

    size_t lower = q->metadata_in;
    size_t unknown = 0u;

    memset(work, 0, sizeof(*work));

    if (q->maybe == 0u) {
        work->total.lower = lower;
        return valid_interval(lower, 0u, q->truth);
    }

    /* First fetch phase. */
    for (size_t j = 0; j < q->maybe; ++j) {
        size_t b = q->ids[j];

        current_level[j] = pred ? predict_initial_level(q, b, divisor) : 0u;

        size_t level = current_level[j];

        for (size_t l = 0; l <= level; ++l)
            ++work->decoder_calls[l];

        ++work->decision_calls[level];

        current[j] = decide_block(b, level, q->threshold);

        lower += current[j].in;
        unknown += current[j].unknown;

        work->total.bytes += LEVEL_BYTES * (level + 1u);

        ++work->total.reads;
    }

    work->total.rounds = 1u;

    if (!valid_interval(lower, unknown, q->truth))
        return 0;

    /* Subsequent dependent refinement phases. */
    while (!meets_tolerance(lower, unknown, divisor)) {
        unsigned char selected[BLOCKS] = {0};
        size_t batch[MAX_BATCH];
        size_t n = 0u;

        for (size_t slot = 0; slot < 64u; ++slot) {
            size_t best = q->maybe;

            for (size_t j = 0; j < q->maybe; ++j) {
                if (selected[j] || current_level[j] + 1u >= LEVELS ||
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

        if (n == 0u)
            return 0;

        for (size_t k = 0; k < n; ++k) {
            size_t j = batch[k];
            size_t b = q->ids[j];

            Decision before = current[j];

            ++current_level[j];

            size_t level = current_level[j];

            Decision after = decide_block(b, level, q->threshold);

            if (after.in < before.in || after.unknown > before.unknown) {
                return 0;
            }

            lower += after.in - before.in;
            unknown -= before.unknown - after.unknown;

            current[j] = after;

            ++work->decoder_calls[level];
            ++work->decision_calls[level];
        }

        work->total.bytes += n * LEVEL_BYTES;
        work->total.reads += n;
        ++work->total.rounds;

        if (!valid_interval(lower, unknown, q->truth))
            return 0;
    }

    work->total.lower = lower;
    work->total.unknown = unknown;
    return 1;
}

static int check_work(
    const WorkTrace* work,
    const StrategyResult* reference,
    const Query* q,
    size_t divisor,
    const char* name
) {
    if (work->total.bytes != reference->bytes ||
        work->total.reads != reference->reads ||
        work->total.rounds != reference->rounds ||
        work->total.lower != reference->lower ||
        work->total.unknown != reference->unknown ||
        !valid_interval(work->total.lower, work->total.unknown, q->truth) ||
        !meets_tolerance(work->total.lower, work->total.unknown, divisor)) {

        fprintf(stderr, "FAIL: CPU work trace differs from %s\n", name);
        return 0;
    }

    return 1;
}

static double estimated_work_us(const WorkTrace* work, const CpuMeasure* m) {
    double total_ns = 0.0;

    for (size_t level = 0; level < LEVELS; ++level) {
        total_ns += (double)work->decoder_calls[level] * m->prefix_ns[level];

        total_ns += (double)work->decision_calls[level] * m->certified_ns[level];
    }

    return total_ns / 1000.0;
}

static double estimated_full8_us(const Query* q, const CpuMeasure* m) {
    return (double)q->maybe * (m->production_full8_ns + m->full8_scan_ns) / 1000.0;
}

static void print_primitives(int family, int pc, int qc) {
    double prefix[LEVELS][SEEDS];
    double certified[LEVELS][SEEDS];

    double production[SEEDS];
    double full_scan[SEEDS];

    for (int s = 0; s < SEEDS; ++s) {
        const CpuMeasure* m = &measurements[family][pc][qc][s];

        for (size_t level = 0; level < LEVELS; ++level) {
            prefix[level][s] = m->prefix_ns[level];
            certified[level][s] = m->certified_ns[level];
        }

        production[s] = m->production_full8_ns;
        full_scan[s] = m->full8_scan_ns;
    }

    printf(
        "pass=%d sel=%2.0f%% | "
        "decode ns/block 2/4/6/8="
        "%.1f/%.1f/%.1f/%.1f | "
        "production8=%.1f\n",

        passes[cpu_pass_indices[pc]],
        selectivities[cpu_query_indices[qc]] * 100.0,

        median5(prefix[0]),
        median5(prefix[1]),
        median5(prefix[2]),
        median5(prefix[3]),

        median5(production)
    );

    printf(
        "                "
        "COUNT ns/block 2/4/6/8="
        "%.1f/%.1f/%.1f/%.1f | "
        "full8 scan=%.1f\n",

        median5(certified[0]),
        median5(certified[1]),
        median5(certified[2]),
        median5(certified[3]),

        median5(full_scan)
    );
}

static void print_query_cpu(int family, int pc, int qc, int tol) {
    printf(
        "pass=%d sel=%2.0f%% tol=%3.1f%% | "
        "estimated CPU us "
        "Full8=%.2f U=%.2f "
        "B64=%.2f P64=%.2f\n",

        passes[cpu_pass_indices[pc]],
        selectivities[cpu_query_indices[qc]] * 100.0,
        tol == 0 ? 1.0 : 0.1,

        median5(query_cpu_us[family][pc][qc][tol][0]),
        median5(query_cpu_us[family][pc][qc][tol][1]),
        median5(query_cpu_us[family][pc][qc][tol][2]),
        median5(query_cpu_us[family][pc][qc][tol][3])
    );
}

int main(void) {
    const char* family_names[FAMILIES] = {"temperature-like", "wind-like"};

    puts("Oracle G: local CPU microbenchmark");
    puts("Direct timing: ns per MAYBE block.");
    puts("Query CPU: estimated from measured operation costs.");
    puts("Reference: fully decoded STORED 8-bit data.");
    puts("No GCP requests or end-to-end query timing.");
    puts("");

    printf("Current synthetic chunk: %d values, %d blocks\n", VALUES, BLOCKS);

    printf(
        "Current metadata: %zu bytes/chunk "
        "(12-byte header + 2 bytes/block)\n",
        (size_t)12u + 2u * (size_t)BLOCKS
    );

    printf(
        "Hypothetical additional error-bound metadata: "
        "%zu bytes/chunk for three float64 bounds\n",
        3u * sizeof(double)
    );

    puts("Three float32 bounds would use 12 bytes, "
         "but conservative float32 serialization "
         "is NOT implemented or validated here.");

    puts("");

    for (int family = 0; family < FAMILIES; ++family) {
        for (int s = 0; s < SEEDS; ++s) {
            for (int pc = 0; pc < CPU_PASS_CASES; ++pc) {
                int pass_index = cpu_pass_indices[pc];

                if (!prepare_sample(family, seeds[s], passes[pass_index])) {

                    fprintf(stderr, "FAIL: prepare sample\n");
                    return 1;
                }

                for (int qc = 0; qc < CPU_QUERY_CASES; ++qc) {

                    int query_index = cpu_query_indices[qc];
                    Query q;

                    if (!build_query(selectivities[query_index], &q)) {

                        fprintf(stderr, "FAIL: build query\n");
                        return 1;
                    }

                    CpuMeasure* m = &measurements[family][pc][qc][s];

                    if (!measure_query_primitives(&q, m)) {
                        fprintf(
                            stderr,
                            "FAIL: CPU measurement "
                            "family=%d seed=%d "
                            "pass=%d sel=%.2f\n",
                            family,
                            s,
                            passes[pass_index],
                            selectivities[query_index]
                        );
                        return 1;
                    }

                    for (int tol = 0; tol < TOLS; ++tol) {
                        size_t divisor = divisors[tol];

                        StrategyResult reference[POLICIES];

                        if (!evaluate_policies(&q, divisor, reference)) {

                            fprintf(stderr, "FAIL: reference policies\n");
                            return 1;
                        }

                        WorkTrace uniform;
                        WorkTrace b64;
                        WorkTrace p64;

                        if (!work_uniform(&q, divisor, &uniform) ||
                            !work_online(&q, divisor, 0, &b64) ||
                            !work_online(&q, divisor, 1, &p64)) {

                            fprintf(stderr, "FAIL: CPU work tracing\n");
                            return 1;
                        }

                        if (!check_work(
                                &uniform, &reference[POLICY_UNIFORM], &q, divisor, "U"
                            ) ||
                            !check_work(
                                &b64, &reference[POLICY_B64], &q, divisor, "B64"
                            ) ||
                            !check_work(
                                &p64, &reference[POLICY_P64], &q, divisor, "P64"
                            )) {
                            return 1;
                        }

                        query_cpu_us[family][pc][qc][tol][0][s] =
                            estimated_full8_us(&q, m);

                        query_cpu_us[family][pc][qc][tol][1][s] =
                            estimated_work_us(&uniform, m);

                        query_cpu_us[family][pc][qc][tol][2][s] =
                            estimated_work_us(&b64, m);

                        query_cpu_us[family][pc][qc][tol][3][s] =
                            estimated_work_us(&p64, m);
                    }
                }
            }
        }
    }

    for (int family = 0; family < FAMILIES; ++family) {
        printf("\n========== %s ==========\n", family_names[family]);

        puts("\n--- Directly measured primitive costs ---");

        for (int pc = 0; pc < CPU_PASS_CASES; ++pc) {
            for (int qc = 0; qc < CPU_QUERY_CASES; ++qc) {
                print_primitives(family, pc, qc);
            }
        }

        puts("\n--- Estimated complete-query CPU work ---");

        for (int pc = 0; pc < CPU_PASS_CASES; ++pc) {
            for (int qc = 0; qc < CPU_QUERY_CASES; ++qc) {
                for (int tol = 0; tol < TOLS; ++tol) {
                    print_query_cpu(family, pc, qc, tol);
                }
            }
        }
    }

    printf("\nBenchmark checksum: %" PRIu64 "\n", (uint64_t)benchmark_sink);

    puts("ALL CPU MICROBENCHMARK CHECKS PASSED");
    puts("Reminder: query CPU numbers are composed estimates, "
         "not end-to-end measured latency.");

    return 0;
}