
#define _POSIX_C_SOURCE 200809L

#include <zfp.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "core.h"

enum {
    SIDE = 128,
    CHUNK = 32,
    GRID = 4,
    CHUNKS = 64,
    BLOCKS_PER_CHUNK = 512,
    BLOCKS = 32768,
    BLOCK_VALUES = 64,
    VALUES = 2097152,
    CHUNK_VALUES = 32768,
    PAYLOAD = 32768,
    METADATA = 1036,
    PACKED = 33804,
    LEVELS = 4,
    BATCH = 64,
    TRIALS = 3
};

typedef struct {
    double lo, hi;
} Bound;

typedef struct {
    double lo[BLOCK_VALUES];
    double hi[BLOCK_VALUES];
    unsigned char ids[BLOCK_VALUES];
    size_t in, unknown;
    int level;
} State;

typedef struct {
    double threshold, target;
    size_t ids[BLOCKS];
    size_t maybe, base, truth, original_truth;
    size_t in_blocks, out_blocks;
} Query;

typedef struct {
    size_t bytes, reads, rounds;
    size_t lower, unknown;
    double cpu_ms;
} Result;

static float raw[VALUES];
static float sorted[VALUES];
static float chunk_data[CHUNK_VALUES];
static float full[BLOCKS][BLOCK_VALUES];

static unsigned char store[CHUNKS][PACKED];
static Bound bound[BLOCKS];
static double error_bound[CHUNKS][LEVELS];

static State states[BLOCKS];
static size_t heap[BLOCKS];
static size_t heap_count;

static Query query;
static volatile uint64_t sink;

static const double selectivities[] = {
    0.0001, 0.001, 0.01, 0.10, 0.50, 0.90
};

static const size_t divisors[] = {100, 1000};

static void fail(const char *message)
{
    fprintf(stderr, "FAIL: %s\n", message);
    exit(EXIT_FAILURE);
}

static double now_ms(void)
{
    struct timespec ts;

    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts) != 0)
        fail("clock_gettime");

    return (double)ts.tv_sec * 1000.0
         + (double)ts.tv_nsec / 1000000.0;
}

static int float_compare(const void *a, const void *b)
{
    float x = *(const float *)a;
    float y = *(const float *)b;
    return (x > y) - (x < y);
}

static double median3(double a[TRIALS])
{
    for (int i = 0; i < TRIALS; ++i)
        for (int j = i + 1; j < TRIALS; ++j)
            if (a[j] < a[i]) {
                double t = a[i];
                a[i] = a[j];
                a[j] = t;
            }

    return a[TRIALS / 2];
}

static void read_raw(const char *path)
{
    FILE *f = fopen(path, "rb");

    if (!f) {
        perror(path);
        exit(EXIT_FAILURE);
    }

    if (fread(raw, sizeof(float), VALUES, f) != VALUES ||
        fgetc(f) != EOF)
        fail("RAW size must be exactly 8388608 bytes");

    fclose(f);

    for (size_t i = 0; i < VALUES; ++i)
        if (!isfinite(raw[i]))
            fail("RAW contains a non-finite value");
}

static int decode_level(
    size_t block,
    int level,
    float out[BLOCK_VALUES]
)
{
    size_t ch = block / BLOCKS_PER_CHUNK;
    size_t local = block % BLOCKS_PER_CHUNK;

    const unsigned char *src =
        store[ch] + local * BLOCK_VALUES;

    if (level == 3)
        return szfp_decompress_block(
            src, BLOCK_VALUES, 8.0,
            (int)zfp_type_float, 3, out
        ) == SZ_OK;

    unsigned char padded[BLOCK_VALUES] = {0};
    memcpy(padded, src, (size_t)(level + 1) * 16u);

    bitstream *bits = stream_open(padded, BLOCK_VALUES);
    if (!bits)
        return 0;

    zfp_stream *zfp = zfp_stream_open(bits);
    if (!zfp) {
        stream_close(bits);
        return 0;
    }

    zfp_stream_set_rate(zfp, 8.0, zfp_type_float, 3, 0);
    zfp_stream_rewind(zfp);

    size_t used = zfp_decode_block_float_3(zfp, out);

    zfp_stream_close(zfp);
    stream_close(bits);

    return used != 0;
}

static size_t raw_index(
    size_t ch, size_t b, size_t local
)
{
    size_t ct = ch / 16u;
    size_t cy = (ch / 4u) % 4u;
    size_t cx = ch % 4u;

    size_t bz = b / 64u;
    size_t by = (b / 8u) % 8u;
    size_t bx = b % 8u;

    size_t z = local / 16u;
    size_t y = (local / 4u) % 4u;
    size_t x = local % 4u;

    size_t t = ct * 32u + bz * 4u + z;
    size_t lat = cy * 32u + by * 4u + y;
    size_t lon = cx * 32u + bx * 4u + x;

    return (t * SIDE + lat) * SIDE + lon;
}

static void build_store(void)
{
    long double squared_error = 0.0L;
    double maximum_error = 0.0;

    for (size_t ch = 0; ch < CHUNKS; ++ch) {
        size_t ct = ch / 16u;
        size_t cy = (ch / 4u) % 4u;
        size_t cx = ch % 4u;

        for (size_t z = 0; z < CHUNK; ++z) {
            for (size_t y = 0; y < CHUNK; ++y) {
                size_t source =
                    ((ct * CHUNK + z) * SIDE +
                     cy * CHUNK + y) * SIDE +
                    cx * CHUNK;

                size_t destination =
                    (z * CHUNK + y) * CHUNK;

                memcpy(
                    chunk_data + destination,
                    raw + source,
                    CHUNK * sizeof(float)
                );
            }
        }

        size_t packed_size = 0;

        if (szfp_pack_chunk(
                chunk_data, CHUNK, CHUNK, CHUNK,
                8.0, 4,
                store[ch], PACKED,
                &packed_size
            ) != SZ_OK ||
            packed_size != PACKED)
            fail("ERA5 chunk compression");

        const unsigned char *meta =
            store[ch] + PAYLOAD;

        float cmin, cmax, reserved;

        memcpy(&cmin, meta, 4);
        memcpy(&cmax, meta + 4, 4);
        memcpy(&reserved, meta + 8, 4);

        if (!isfinite(cmin) || !isfinite(cmax) ||
            cmin > cmax || reserved != 0.0f)
            fail("chunk metadata header");

        double span = (double)cmax - (double)cmin;
        double margin = span / 255.0;
        const unsigned char *offsets = meta + 12;

        for (size_t b = 0; b < BLOCKS_PER_CHUNK; ++b) {
            size_t id = ch * BLOCKS_PER_CHUNK + b;

            double lo = (double)cmin;
            double hi = (double)cmax;

            if (span != 0.0) {
                lo = (double)cmin +
                    (double)offsets[2 * b] * span / 255.0
                    - margin;

                hi = (double)cmin +
                    (double)offsets[2 * b + 1] *
                    span / 255.0 + margin;
            }

            bound[id].lo = nextafter(lo, -INFINITY);
            bound[id].hi = nextafter(hi, INFINITY);

            if (!decode_level(id, 3, full[id]))
                fail("full block decode");

            for (int l = 0; l < 3; ++l) {
                float prefix[BLOCK_VALUES];

                if (!decode_level(id, l, prefix))
                    fail("prefix block decode");

                for (size_t i = 0; i < BLOCK_VALUES; ++i) {
                    double difference = fabs(
                        (double)prefix[i] -
                        (double)full[id][i]
                    );

                    if (difference > error_bound[ch][l])
                        error_bound[ch][l] = difference;
                }
            }

            for (size_t i = 0; i < BLOCK_VALUES; ++i) {
                float v = full[id][i];

                if (!isfinite(v) ||
                    (double)v < bound[id].lo ||
                    (double)v > bound[id].hi)
                    fail("metadata does not contain stored value");

                double difference = fabs(
                    (double)raw[raw_index(ch, b, i)] -
                    (double)v
                );

                if (difference > maximum_error)
                    maximum_error = difference;

                squared_error +=
                    (long double)difference * difference;
            }
        }

        for (int l = 0; l < 3; ++l)
            if (error_bound[ch][l] > 0.0)
                error_bound[ch][l] =
                    nextafter(error_bound[ch][l], INFINITY);
    }

    printf(
        "RAW=%zu payload=%zu metadata=%zu packed=%zu "
        "ratio=%.3fx\n",
        sizeof(raw),
        (size_t)CHUNKS * PAYLOAD,
        (size_t)CHUNKS * METADATA,
        (size_t)CHUNKS * PACKED,
        (double)sizeof(raw) /
        ((double)CHUNKS * PACKED)
    );

    printf(
        "Original-to-stored max_error=%.9g K "
        "RMSE=%.9Lg K\n",
        maximum_error,
        sqrtl(squared_error / VALUES)
    );

    for (int l = 0; l < 3; ++l) {
        double minimum = INFINITY;
        double maximum = 0.0;
        double sum = 0.0;

        for (size_t ch = 0; ch < CHUNKS; ++ch) {
            double e = error_bound[ch][l];

            if (e < minimum)
                minimum = e;

            if (e > maximum)
                maximum = e;

            sum += e;
        }

        printf(
            "%d-bit prefix error bound (K): "
            "min=%.9g mean=%.9g max=%.9g\n",
            (l + 1) * 2,
            minimum,
            sum / CHUNKS,
            maximum
        );
    }

    printf(
        "Offline float64 prefix bounds: %zu bytes "
        "(NOT stored in current packed format)\n",
        (size_t)CHUNKS * 3u * sizeof(double)
    );

    puts("PACK + METADATA + PREFIX BOUNDS PASSED");
}

static int apply_block(
    size_t id,
    int target,
    double threshold,
    int first
)
{
    State *s = &states[id];

    if (first)
        memset(s, 0, sizeof(*s));

    int begin = first ? 0 : s->level + 1;
    size_t ch = id / BLOCKS_PER_CHUNK;

    for (int l = begin; l <= target; ++l) {
        float values[BLOCK_VALUES];

        if (!decode_level(id, l, values))
            return 0;

        double eps = error_bound[ch][l];

        if (first && l == 0) {
            for (size_t i = 0; i < BLOCK_VALUES; ++i) {
                double v = (double)values[i];

                double lo = eps == 0.0
                    ? v : nextafter(v - eps, -INFINITY);

                double hi = eps == 0.0
                    ? v : nextafter(v + eps, INFINITY);

                s->lo[i] = lo;
                s->hi[i] = hi;

                if (lo > threshold) {
                    ++s->in;
                } else if (hi > threshold) {
                    s->ids[s->unknown++] =
                        (unsigned char)i;
                }
            }
        } else {
            size_t remaining = 0;
            size_t previous = s->unknown;

            for (size_t k = 0; k < previous; ++k) {
                size_t i = s->ids[k];
                double v = (double)values[i];

                double lo = eps == 0.0
                    ? v : nextafter(v - eps, -INFINITY);

                double hi = eps == 0.0
                    ? v : nextafter(v + eps, INFINITY);

                if (lo < s->lo[i])
                    lo = s->lo[i];

                if (hi > s->hi[i])
                    hi = s->hi[i];

                if (lo > hi)
                    return 0;

                s->lo[i] = lo;
                s->hi[i] = hi;

                if (lo > threshold) {
                    ++s->in;
                } else if (hi > threshold) {
                    s->ids[remaining++] =
                        (unsigned char)i;
                }
            }

            s->unknown = remaining;
        }

        s->level = l;
    }

    return 1;
}

static int meets(
    size_t lower,
    size_t unknown,
    size_t divisor
)
{
    return unknown <= lower / divisor;
}

static void check_result(
    const Query *q,
    const Result *r,
    size_t divisor
)
{
    if (r->lower > q->truth ||
        r->unknown > VALUES - r->lower ||
        q->truth > r->lower + r->unknown ||
        !meets(r->lower, r->unknown, divisor) ||
        r->bytes > q->maybe * BLOCK_VALUES)
        fail("certified COUNT or payload check");
}

static void build_query(double selectivity)
{
    memset(&query, 0, sizeof(query));

    query.target = selectivity;

    size_t position =
        (size_t)((1.0 - selectivity) * VALUES);

    if (position >= VALUES)
        position = VALUES - 1;

    query.threshold = (double)sorted[position];

    for (size_t id = 0; id < BLOCKS; ++id) {
        size_t actual_in = 0;

        for (size_t i = 0; i < BLOCK_VALUES; ++i)
            actual_in +=
                (double)full[id][i] > query.threshold;

        query.truth += actual_in;

        if (bound[id].lo > query.threshold) {
            if (actual_in != BLOCK_VALUES)
                fail("metadata false IN");

            query.base += BLOCK_VALUES;
            ++query.in_blocks;

        } else if (bound[id].hi <= query.threshold) {
            if (actual_in != 0)
                fail("metadata false OUT");

            ++query.out_blocks;

        } else {
            query.ids[query.maybe++] = id;
        }
    }

    for (size_t i = 0; i < VALUES; ++i)
        query.original_truth +=
            (double)raw[i] > query.threshold;

    if (query.in_blocks + query.out_blocks +
        query.maybe != BLOCKS)
        fail("metadata block accounting");
}

static int better(size_t a, size_t b)
{
    if (states[a].unknown != states[b].unknown)
        return states[a].unknown > states[b].unknown;

    return a < b;
}

static void heap_push(size_t id)
{
    size_t p = heap_count++;

    while (p > 0) {
        size_t parent = (p - 1u) / 2u;

        if (!better(id, heap[parent]))
            break;

        heap[p] = heap[parent];
        p = parent;
    }

    heap[p] = id;
}

static size_t heap_pop(void)
{
    size_t best = heap[0];
    size_t last = heap[--heap_count];

    if (heap_count == 0)
        return best;

    size_t p = 0;

    while (2u * p + 1u < heap_count) {
        size_t child = 2u * p + 1u;

        if (child + 1u < heap_count &&
            better(heap[child + 1u], heap[child]))
            ++child;

        if (!better(heap[child], last))
            break;

        heap[p] = heap[child];
        p = child;
    }

    heap[p] = last;
    return best;
}

static Result full8_query(const Query *q)
{
    Result r = {0};
    float decoded[BLOCK_VALUES];

    r.lower = q->base;
    r.bytes = q->maybe * BLOCK_VALUES;
    r.reads = q->maybe;
    r.rounds = q->maybe != 0;

    for (size_t j = 0; j < q->maybe; ++j) {
        size_t id = q->ids[j];

        if (!decode_level(id, 3, decoded))
            fail("Full8 query decode");

        for (size_t i = 0; i < BLOCK_VALUES; ++i)
            r.lower +=
                (double)decoded[i] > q->threshold;
    }

    if (r.lower != q->truth)
        fail("Full8 exact COUNT");

    return r;
}

static Result uniform_query(
    const Query *q,
    size_t divisor
)
{
    Result r = {0};

    for (int level = 0; level < LEVELS; ++level) {
        size_t lower = q->base;
        size_t unknown = 0;

        for (size_t j = 0; j < q->maybe; ++j) {
            size_t id = q->ids[j];

            if (!apply_block(
                    id, level, q->threshold,
                    level == 0
                ))
                fail("uniform prefix");

            lower += states[id].in;
            unknown += states[id].unknown;
        }

        if (q->maybe != 0) {
            r.bytes += q->maybe * 16u;
            r.reads += q->maybe;
            ++r.rounds;
        }

        r.lower = lower;
        r.unknown = unknown;

        if (lower > q->truth ||
            q->truth > lower + unknown)
            fail("uniform interval");

        if (meets(lower, unknown, divisor))
            return r;
    }

    fail("uniform did not converge");
    return r;
}

static int predict_level(size_t id)
{
    double width = bound[id].hi - bound[id].lo;
    size_t ch = id / BLOCKS_PER_CHUNK;

    if (width <= 0.0)
        return 2;

    if (error_bound[ch][0] <= width / 128.0)
        return 0;

    if (error_bound[ch][1] <= width / 128.0)
        return 1;

    return 2;
}

static Result online_query(
    const Query *q,
    size_t divisor,
    int predicted
)
{
    Result r = {0};
    size_t lower = q->base;
    size_t unknown = 0;
    size_t initial_bytes = 0;

    heap_count = 0;

    if (q->maybe == 0) {
        r.lower = lower;
        return r;
    }

    for (size_t j = 0; j < q->maybe; ++j) {
        size_t id = q->ids[j];
        int level = predicted ? predict_level(id) : 0;

        if (!apply_block(id, level, q->threshold, 1))
            fail("online initial prefix");

        lower += states[id].in;
        unknown += states[id].unknown;

        initial_bytes += (size_t)(level + 1) * 16u;

        if (states[id].unknown > 0 &&
            states[id].level < 3)
            heap_push(id);
    }

    r.bytes = initial_bytes;
    r.reads = q->maybe;
    r.rounds = 1;

    while (!meets(lower, unknown, divisor)) {
        size_t batch[BATCH];
        size_t count = 0;

        while (count < BATCH && heap_count > 0)
            batch[count++] = heap_pop();

        if (count == 0)
            fail("online cannot refine");

        for (size_t k = 0; k < count; ++k) {
            size_t id = batch[k];

            size_t old_in = states[id].in;
            size_t old_unknown = states[id].unknown;
            int next_level = states[id].level + 1;

            if (!apply_block(
                    id, next_level, q->threshold, 0
                ))
                fail("online refinement");

            if (states[id].in < old_in ||
                states[id].unknown > old_unknown)
                fail("nonmonotone interval");

            lower += states[id].in - old_in;
            unknown -=
                old_unknown - states[id].unknown;

            if (states[id].unknown > 0 &&
                states[id].level < 3)
                heap_push(id);
        }

        r.bytes += count * 16u;
        r.reads += count;
        ++r.rounds;

        if (lower > q->truth ||
            q->truth > lower + unknown)
            fail("online COUNT interval");
    }

    r.lower = lower;
    r.unknown = unknown;
    return r;
}

static Result exact_oracle(const Query *q)
{
    Result r = {0};
    r.lower = q->truth;

    for (size_t j = 0; j < q->maybe; ++j) {
        size_t id = q->ids[j];
        int resolved = 0;

        for (int level = 0; level < LEVELS; ++level) {
            if (!apply_block(
                    id, level, q->threshold,
                    level == 0
                ))
                fail("offline oracle prefix");

            if (states[id].unknown == 0) {
                r.bytes += (size_t)(level + 1) * 16u;
                resolved = 1;
                break;
            }
        }

        if (!resolved)
            fail("offline oracle unresolved at 8-bit");
    }

    return r;
}

static Result run_policy(
    int policy,
    const Query *q,
    size_t divisor
)
{
    if (policy == 0)
        return full8_query(q);

    if (policy == 1)
        return uniform_query(q, divisor);

    if (policy == 2)
        return online_query(q, divisor, 0);

    return online_query(q, divisor, 1);
}

static Result benchmark(
    int policy,
    const Query *q,
    size_t divisor
)
{
    double times[TRIALS];
    Result first = {0};

    for (int t = 0; t < TRIALS; ++t) {
        double start = now_ms();
        Result r = run_policy(policy, q, divisor);
        double finish = now_ms();

        check_result(q, &r, divisor);

        if (t == 0) {
            first = r;
        } else if (
            r.bytes != first.bytes ||
            r.reads != first.reads ||
            r.rounds != first.rounds ||
            r.lower != first.lower ||
            r.unknown != first.unknown
        ) {
            fail("non-reproducible policy result");
        }

        times[t] = finish - start;

        sink +=
            (uint64_t)r.lower +
            (uint64_t)r.unknown +
            (uint64_t)r.bytes;
    }

    first.cpu_ms = median3(times);
    return first;
}

static void print_policy(
    const char *name,
    const Result *r,
    size_t full_bytes
)
{
    double ratio = r->bytes == 0
        ? 1.0
        : (double)full_bytes / (double)r->bytes;

    printf(
        "  %-6s bytes=%7zu saved=%6.2fx "
        "reads=%6zu rounds=%5zu "
        "COUNT=[%zu,%zu] CPU=%.3f ms\n",
        name,
        r->bytes,
        ratio,
        r->reads,
        r->rounds,
        r->lower,
        r->lower + r->unknown,
        r->cpu_ms
    );
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

    uint16_t endian = 1;

    if (*(unsigned char *)&endian != 1 ||
        sizeof(float) != 4)
        fail("requires little-endian float32");

    read_raw(path);

    printf("ERA5 128^3: %s\n", path);
    printf(
        "Shape=128x128x128 chunks=%d blocks=%d\n",
        CHUNKS, BLOCKS
    );

    build_store();

    memcpy(sorted, full, sizeof(sorted));
    qsort(sorted, VALUES, sizeof(float), float_compare);

    for (size_t si = 0;
         si < sizeof(selectivities) / sizeof(selectivities[0]);
         ++si) {

        build_query(selectivities[si]);

        double actual =
            100.0 * (double)query.truth / VALUES;

        printf(
            "\n=== selectivity target=%.2f%% "
            "actual=%.4f%% threshold=%.9g K ===\n",
            query.target * 100.0,
            actual,
            query.threshold
        );

        printf(
            "blocks IN=%zu OUT=%zu MAYBE=%zu "
            "stored_COUNT=%zu original_COUNT=%zu\n",
            query.in_blocks,
            query.out_blocks,
            query.maybe,
            query.truth,
            query.original_truth
        );

        Result oracle = exact_oracle(&query);

        for (size_t ti = 0; ti < 2; ++ti) {
            size_t divisor = divisors[ti];

            Result full8 = benchmark(
                0, &query, divisor
            );

            Result uniform = benchmark(
                1, &query, divisor
            );

            Result b64 = benchmark(
                2, &query, divisor
            );

            Result p64 = benchmark(
                3, &query, divisor
            );

            if (oracle.bytes > full8.bytes)
                fail("offline oracle exceeds Full8");

            printf(
                "\ntolerance=%.1f%% "
                "offline_exact_bytes=%zu "
                "offline_saving=%.2fx\n",
                100.0 / (double)divisor,
                oracle.bytes,
                oracle.bytes == 0
                    ? 1.0
                    : (double)full8.bytes /
                      (double)oracle.bytes
            );

            print_policy(
                "Full8", &full8, full8.bytes
            );

            print_policy(
                "Uniform", &uniform, full8.bytes
            );

            print_policy(
                "B64", &b64, full8.bytes
            );

            print_policy(
                "P64*", &p64, full8.bytes
            );
        }
    }

    printf("\nBenchmark checksum: %llu\n",
           (unsigned long long)sink);

    puts("ALL ERA5 MULTI-CHUNK ORACLE CHECKS PASSED");
    puts("CPU = local exploratory measurement, NOT cloud latency.");
    puts("P64* = new metadata-only heuristic, not old P64.");
    puts("Offline exact oracle uses hindsight.");

    return 0;
}