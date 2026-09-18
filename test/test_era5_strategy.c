
#define _POSIX_C_SOURCE 200809L

#define main era5_batch_embedded_main
#include "test_era5_batch.c"
#undef main

enum {
    A_POLICIES = 10,
    A_SCENARIOS = 4,
    A_TRIALS = 3,
    A_PARALLEL = 64
};

typedef struct {
    const char *name;
    size_t batch;
    int predicted;
    size_t cap;
} APolicy;

typedef struct {
    Result r;
    size_t waves;
} ATrace;

typedef struct {
    const char *name;
    double latency_ms;
    double bandwidth_mb_s;
} AScenario;

static const APolicy policies[A_POLICIES] = {
    {"Scan",       0,    0, 0},
    {"Full8",      0,    0, 0},
    {"B64",       64,    0, 0},
    {"B256",     256,    0, 0},
    {"B1024",   1024,    0, 0},
    {"B4096",   4096,    0, 0},
    {"P1024*",  1024,    1, 0},
    {"B1024-C2",1024,    0, 2},
    {"P1024-C2",1024,    1, 2},
    {"P1024-C4",1024,    1, 4}
};

static const AScenario scenarios[A_SCENARIOS] = {
    {"SLOW_LOW_RTT",  0.05,  0.25},
    {"SLOW_MID_RTT",  5.00,  0.25},
    {"FAST_LOW_RTT",  0.05, 50.00},
    {"FAST_HIGH_RTT",50.00, 50.00}
};

static volatile uint64_t a_sink = 0;

static size_t a_wave_count(size_t reads)
{
    return reads / A_PARALLEL +
           (reads % A_PARALLEL != 0);
}

static void a_record(ATrace *t, size_t bytes, size_t reads)
{
    if (bytes == 0 || reads == 0)
        fail("invalid fetch phase");

    t->r.bytes += bytes;
    t->r.reads += reads;
    ++t->r.rounds;
    t->waves += a_wave_count(reads);
}

static void a_interval(
    const Query *q,
    size_t lower,
    size_t unknown
)
{
    if (lower > q->truth ||
        unknown > VALUES - lower ||
        q->truth > lower + unknown)
        fail("incorrect intermediate COUNT interval");
}

static int a_scan(const Query *q, ATrace *t)
{
    float values[BLOCK_VALUES];
    size_t count = 0;

    memset(t, 0, sizeof(*t));

    for (size_t id = 0; id < BLOCKS; ++id) {
        if (!decode_level(id, 3, values))
            return 0;

        for (size_t i = 0; i < BLOCK_VALUES; ++i)
            count += (double)values[i] > q->threshold;
    }

    a_record(t, (size_t)CHUNKS * PAYLOAD, CHUNKS);

    t->r.lower = count;

    return count == q->truth;
}

static int a_full8(const Query *q, ATrace *t)
{
    memset(t, 0, sizeof(*t));

    t->r = full8_query(q);

    if (q->maybe != 0)
        t->waves = a_wave_count(q->maybe);

    return t->r.lower == q->truth &&
           t->r.unknown == 0 &&
           t->r.bytes == q->maybe * BLOCK_VALUES &&
           t->r.rounds == (q->maybe != 0);
}

static int a_finish_exact(
    const Query *q,
    ATrace *t,
    size_t *lower,
    size_t *unknown
)
{
    float values[BLOCK_VALUES];
    size_t bytes = 0;
    size_t reads = 0;

    for (size_t j = 0; j < q->maybe; ++j) {
        size_t id = q->ids[j];
        State *s = &states[id];

        if (s->unknown == 0)
            continue;

        if (s->level < 0 || s->level >= 3)
            return 0;

        if (!decode_level(id, 3, values))
            return 0;

        size_t exact = 0;

        for (size_t i = 0; i < BLOCK_VALUES; ++i)
            exact += (double)values[i] > q->threshold;

        if (exact < s->in ||
            exact - s->in > s->unknown)
            return 0;

        *lower += exact - s->in;
        *unknown -= s->unknown;

        bytes += (size_t)(3 - s->level) * 16u;
        ++reads;
    }

    if (reads == 0)
        return 0;

    a_record(t, bytes, reads);
    a_interval(q, *lower, *unknown);

    t->r.lower = *lower;
    t->r.unknown = *unknown;

    return *unknown == 0 && *lower == q->truth;
}

static int a_online(
    const Query *q,
    size_t divisor,
    const APolicy *p,
    ATrace *t
)
{
    size_t lower = q->base;
    size_t unknown = 0;
    size_t initial_bytes = 0;

    memset(t, 0, sizeof(*t));
    heap_count = 0;

    if (p->batch == 0 || p->batch > 4096)
        return 0;

    if (q->maybe == 0) {
        t->r.lower = lower;
        return lower == q->truth;
    }

    for (size_t j = 0; j < q->maybe; ++j) {
        size_t id = q->ids[j];
        int level = p->predicted ? predict_level(id) : 0;

        if (!apply_block(id, level, q->threshold, 1))
            return 0;

        lower += states[id].in;
        unknown += states[id].unknown;

        initial_bytes += (size_t)(level + 1) * 16u;

        if (states[id].unknown != 0 &&
            states[id].level < 3)
            heap_push(id);
    }

    a_record(t, initial_bytes, q->maybe);
    a_interval(q, lower, unknown);

    while (!meets(lower, unknown, divisor)) {
        if (p->cap != 0 && t->r.rounds >= p->cap) {
            return a_finish_exact(
                q, t, &lower, &unknown
            );
        }

        size_t batch[4096];
        size_t n = 0;

        while (n < p->batch && heap_count != 0)
            batch[n++] = heap_pop();

        if (n == 0)
            return 0;

        for (size_t k = 0; k < n; ++k) {
            size_t id = batch[k];
            State *s = &states[id];

            size_t before_in = s->in;
            size_t before_unknown = s->unknown;

            if (!apply_block(
                    id,
                    s->level + 1,
                    q->threshold,
                    0
                ))
                return 0;

            if (s->in < before_in ||
                s->unknown > before_unknown)
                return 0;

            lower += s->in - before_in;
            unknown -= before_unknown - s->unknown;

            if (s->unknown != 0 && s->level < 3)
                heap_push(id);
        }

        a_record(t, n * 16u, n);
        a_interval(q, lower, unknown);
    }

    t->r.lower = lower;
    t->r.unknown = unknown;

    return 1;
}

static int a_execute(
    const Query *q,
    size_t divisor,
    int policy,
    ATrace *t
)
{
    if (policy == 0)
        return a_scan(q, t);

    if (policy == 1)
        return a_full8(q, t);

    return a_online(q, divisor, &policies[policy], t);
}

static int a_validate(
    const Query *q,
    size_t divisor,
    int policy,
    const ATrace *t
)
{
    const Result *r = &t->r;

    a_interval(q, r->lower, r->unknown);

    if (!meets(r->lower, r->unknown, divisor))
        return 0;

    if (policy == 0) {
        return r->lower == q->truth &&
               r->unknown == 0 &&
               r->bytes == (size_t)CHUNKS * PAYLOAD &&
               r->reads == CHUNKS &&
               r->rounds == 1 &&
               t->waves == 1;
    }

    if (r->bytes > q->maybe * BLOCK_VALUES)
        return 0;

    if (policy == 1) {
        return r->lower == q->truth &&
               r->unknown == 0 &&
               r->bytes == q->maybe * BLOCK_VALUES &&
               r->reads == q->maybe &&
               r->rounds == (q->maybe != 0);
    }

    if (policies[policy].cap != 0 &&
        r->rounds > policies[policy].cap + 1)
        return 0;

    return r->rounds > 0 || q->maybe == 0;
}

static int a_measure(
    const Query *q,
    size_t divisor,
    int policy,
    ATrace *out
)
{
    double times[A_TRIALS];

    if (!a_execute(q, divisor, policy, out) ||
        !a_validate(q, divisor, policy, out))
        return 0;

    for (int trial = 0; trial < A_TRIALS; ++trial) {
        ATrace current;

        double start = now_ms();

        if (!a_execute(q, divisor, policy, &current))
            return 0;

        double end = now_ms();

        if (!a_validate(q, divisor, policy, &current))
            return 0;

        if (current.r.bytes != out->r.bytes ||
            current.r.reads != out->r.reads ||
            current.r.rounds != out->r.rounds ||
            current.r.lower != out->r.lower ||
            current.r.unknown != out->r.unknown ||
            current.waves != out->waves)
            return 0;

        times[trial] = end - start;

        a_sink += (uint64_t)current.r.lower +
                  (uint64_t)current.r.bytes +
                  (uint64_t)current.r.rounds;
    }

    out->r.cpu_ms = median3(times);

    return isfinite(out->r.cpu_ms) &&
           out->r.cpu_ms >= 0;
}

static double a_model(
    const ATrace *t,
    int policy,
    const AScenario *s,
    int metadata_cold
)
{
    double bytes = (double)t->r.bytes;
    size_t rounds = t->r.rounds;
    size_t waves = t->waves;

    if (metadata_cold && policy != 0) {
        bytes += (double)CHUNKS * METADATA;

        if (policy >= 2)
            bytes += (double)CHUNKS * 3u * sizeof(double);

        ++rounds;
        ++waves;
    }

    return t->r.cpu_ms +
           (double)rounds * s->latency_ms +
           (double)waves * 0.02 +
           bytes / (1000.0 * s->bandwidth_mb_s);
}

static void a_print_models(
    const ATrace t[A_POLICIES]
)
{
    for (int si = 0; si < A_SCENARIOS; ++si) {
        const AScenario *s = &scenarios[si];

        for (int cold = 0; cold <= 1; ++cold) {
            int best = 0;
            double best_ms = INFINITY;

            for (int p = 0; p < A_POLICIES; ++p) {
                double ms = a_model(
                    &t[p], p, s, cold
                );

                if (ms < best_ms) {
                    best_ms = ms;
                    best = p;
                }
            }

            printf(
                "  MODEL %-13s metadata=%-4s | "
                "Scan=%.3f Full8=%.3f "
                "B1024=%.3f P1024*=%.3f "
                "B-C2=%.3f P-C2=%.3f | "
                "MODEL_MIN=%s\n",
                s->name,
                cold ? "cold" : "warm",
                a_model(&t[0], 0, s, cold),
                a_model(&t[1], 1, s, cold),
                a_model(&t[4], 4, s, cold),
                a_model(&t[6], 6, s, cold),
                a_model(&t[7], 7, s, cold),
                a_model(&t[8], 8, s, cold),
                policies[best].name
            );
        }
    }
}

int main(int argc, char **argv)
{
    const char *path = argc > 1
        ? argv[1]
        : "../data/era5_t2m_128cube_f32le.raw";

    if (argc > 2) {
        fprintf(stderr, "Usage: %s [RAW_path]\n", argv[0]);
        return 1;
    }

    read_raw(path);
    build_store();

    memcpy(sorted, full, sizeof(sorted));
    qsort(sorted, VALUES, sizeof(float), float_compare);

    puts("\nERA5 strategy comparison");
    puts("Scan = 64 whole-chunk payload reads, no index.");
    puts("Full8 = metadata-filtered 64-byte block reads.");
    puts("Prefix = 16-byte logical refinement reads.");
    puts("C2/C4 = at most 2/4 partial rounds, then exact fallback.");
    puts("P1024* uses the ERA5 test's metadata-only predictor.");
    puts("CPU is local; network and metadata fetches are modeled.");
    puts("Cold metadata assumes 64 parallel chunk-index reads.");
    puts("Prefix error bounds are computed offline at ingest.");
    puts("");

    for (size_t si = 0;
         si < sizeof(selectivities) / sizeof(selectivities[0]);
         ++si) {

        build_query(selectivities[si]);

        printf(
            "\n=== target=%.2f%% actual=%.4f%% "
            "threshold=%.9g K MAYBE=%zu "
            "stored_COUNT=%zu original_COUNT=%zu ===\n",
            selectivities[si] * 100.0,
            100.0 * (double)query.truth / VALUES,
            query.threshold,
            query.maybe,
            query.truth,
            query.original_truth
        );

        for (size_t ti = 0;
             ti < sizeof(divisors) / sizeof(divisors[0]);
             ++ti) {

            size_t divisor = divisors[ti];
            ATrace results[A_POLICIES];

            printf(
                "\n--- COUNT tolerance %.1f%% ---\n",
                100.0 / (double)divisor
            );

            for (int p = 0; p < A_POLICIES; ++p) {
                if (!a_measure(
                        &query, divisor, p,
                        &results[p]
                    )) {
                    fprintf(
                        stderr,
                        "FAIL: strategy=%s sel=%.4f "
                        "divisor=%zu\n",
                        policies[p].name,
                        selectivities[si],
                        divisor
                    );
                    return 1;
                }

                const Result *r = &results[p].r;

                printf(
                    "  %-10s bytes=%7zu "
                    "reads=%6zu rounds=%5zu "
                    "waves=%5zu COUNT=[%zu,%zu] "
                    "CPU=%.3f ms\n",
                    policies[p].name,
                    r->bytes,
                    r->reads,
                    r->rounds,
                    results[p].waves,
                    r->lower,
                    r->lower + r->unknown,
                    r->cpu_ms
                );
            }

            Result old_b64 = online_query(
                &query, divisor, 0
            );

            const Result *new_b64 = &results[2].r;

            if (old_b64.bytes != new_b64->bytes ||
                old_b64.reads != new_b64->reads ||
                old_b64.rounds != new_b64->rounds ||
                old_b64.lower != new_b64->lower ||
                old_b64.unknown != new_b64->unknown)
                fail("B64 differs from previous ERA5 test");

            a_print_models(results);
        }
    }

    printf(
        "\nBenchmark checksum: %llu\n",
        (unsigned long long)a_sink
    );

    puts("ALL ERA5 STRATEGY CHECKS PASSED");
    puts("Model time is NOT measured cloud latency.");

    return 0;
}