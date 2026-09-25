#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

typedef int (*decompress_fn)(
    const void* src,
    void* dest,
    size_t destsize,
    int nthreads
);

static size_t count_gt(const float* v, size_t n, double th) {
    size_t c = 0;
    for (size_t i = 0; i < n; i++) {
        c += (double)v[i] > th;
    }
    return c;
}

int bc_count_gt(
    void* fn,
    const unsigned char* buf,
    size_t buf_size,
    const uint64_t* offs,
    const uint64_t* sizes,
    size_t n,
    size_t raw_nbytes,
    double th,
    size_t* out
) {
    decompress_fn dec = (decompress_fn)fn;
    float* tmp = (float*)malloc(raw_nbytes);
    size_t c = 0;

    if (tmp == NULL) {
        return 4;
    }
    for (size_t i = 0; i < n; i++) {
        if (offs[i] > buf_size || buf_size - offs[i] < sizes[i]) {
            free(tmp);
            return 3;
        }
        if (dec(buf + offs[i], tmp, raw_nbytes, 1) != (int)raw_nbytes) {
            free(tmp);
            return 5;
        }
        c += count_gt(tmp, raw_nbytes / sizeof(float), th);
    }
    free(tmp);
    *out = c;
    return 0;
}

size_t bc_count_f32(const float* v, size_t n, double th) {
    return count_gt(v, n, th);
}

int bc_decode_many(
    void* fn,
    const unsigned char* buf,
    size_t buf_size,
    const uint64_t* offs,
    const uint64_t* sizes,
    size_t n,
    size_t raw_nbytes,
    unsigned char* out
) {
    decompress_fn dec = (decompress_fn)fn;
    for (size_t i = 0; i < n; i++) {
        if (offs[i] > buf_size || buf_size - offs[i] < sizes[i]) {
            return 3;
        }
        if (dec(buf + offs[i], out + i * raw_nbytes, raw_nbytes, 1) !=
            (int)raw_nbytes) {
            return 5;
        }
    }
    return 0;
}

int bc_count_abs_gt(
    void* fn,
    const unsigned char* buf,
    size_t buf_size,
    const uint64_t* offs,
    const uint64_t* sizes,
    size_t n,
    size_t raw_nbytes,
    double th,
    size_t* out
) {
    decompress_fn dec = (decompress_fn)fn;
    float* tmp = (float*)malloc(raw_nbytes);
    size_t c = 0;

    if (tmp == NULL) {
        return 4;
    }
    for (size_t i = 0; i < n; i++) {
        if (offs[i] > buf_size || buf_size - offs[i] < sizes[i]) {
            free(tmp);
            return 3;
        }
        if (dec(buf + offs[i], tmp, raw_nbytes, 1) != (int)raw_nbytes) {
            free(tmp);
            return 5;
        }
        for (size_t k = 0; k < raw_nbytes / sizeof(float); k++) {
            double v = tmp[k];
            c += (v > th) || (v < -th);
        }
    }
    free(tmp);
    *out = c;
    return 0;
}
