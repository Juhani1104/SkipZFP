
#include <zfp.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../core.h"

enum {
    SIDE = 32,
    BLOCK_SIDE = 4,
    BLOCK_VALUES = 64,
    BLOCKS_PER_AXIS = 8,
    BLOCKS = 512,
    VALUES = 32768,
    PAYLOAD_BYTES = 32768,
    METADATA_BYTES = 12 + BLOCKS * 2,
    PACKED_BYTES = PAYLOAD_BYTES + METADATA_BYTES
};

static float source[VALUES];
static float reconstructed[VALUES];
static unsigned char packed[PACKED_BYTES];

static int read_raw(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror(path);
        return 0;
    }

    size_t n = fread(source, sizeof(float), VALUES, f);
    int extra = fgetc(f);
    fclose(f);

    if (n != VALUES || extra != EOF) {
        fprintf(stderr, "FAIL: RAW file size is not 131072 bytes\n");
        return 0;
    }

    for (size_t i = 0; i < VALUES; ++i) {
        if (!isfinite(source[i])) {
            fprintf(stderr, "FAIL: non-finite RAW value at %zu\n", i);
            return 0;
        }
    }

    return 1;
}

static int unpack_whole(void)
{
    bitstream *bits = stream_open(packed, PAYLOAD_BYTES);
    if (!bits)
        return 0;

    zfp_stream *zfp = zfp_stream_open(bits);
    if (!zfp) {
        stream_close(bits);
        return 0;
    }

    zfp_stream_set_rate(zfp, 8.0, zfp_type_float, 3, 0);
    zfp_stream_rewind(zfp);

    zfp_field *field = zfp_field_3d(
        reconstructed, zfp_type_float, SIDE, SIDE, SIDE
    );

    if (!field) {
        zfp_stream_close(zfp);
        stream_close(bits);
        return 0;
    }

    size_t result = zfp_decompress(zfp, field);

    zfp_field_free(field);
    zfp_stream_close(zfp);
    stream_close(bits);

    return result != 0;
}

static int check_blocks_and_metadata(void)
{
    const unsigned char *meta = packed + PAYLOAD_BYTES;

    float cmin, cmax, eps;
    memcpy(&cmin, meta, sizeof(float));
    memcpy(&cmax, meta + 4, sizeof(float));
    memcpy(&eps, meta + 8, sizeof(float));

    if (!isfinite(cmin) || !isfinite(cmax) ||
        !isfinite(eps) || cmin > cmax || eps != 0.0f) {
        fprintf(stderr, "FAIL: invalid metadata header\n");
        return 0;
    }

    const unsigned char *offsets = meta + 12;
    double span = (double)cmax - (double)cmin;
    double delta = span == 0.0 ? 0.0 : span / 255.0;

    size_t checked_values = 0;
    double actual_min = INFINITY;
    double actual_max = -INFINITY;

    for (size_t b = 0; b < BLOCKS; ++b) {
        float block[BLOCK_VALUES];

        SzResult status = szfp_decompress_block(
            packed + b * BLOCK_VALUES,
            BLOCK_VALUES,
            8.0,
            (int)zfp_type_float,
            3,
            block
        );

        if (status != SZ_OK) {
            fprintf(stderr, "FAIL: block decode b=%zu code=%d\n",
                    b, (int)status);
            return 0;
        }

        size_t bz = b / (BLOCKS_PER_AXIS * BLOCKS_PER_AXIS);
        size_t by = (b / BLOCKS_PER_AXIS) % BLOCKS_PER_AXIS;
        size_t bx = b % BLOCKS_PER_AXIS;

        double lower = (double)cmin;
        double upper = (double)cmax;

        if (span != 0.0) {
            lower += ((double)offsets[2 * b] / 255.0) * span
                     - delta;

            upper = (double)cmin
                  + ((double)offsets[2 * b + 1] / 255.0) * span
                  + delta;
        }

        for (size_t z = 0; z < BLOCK_SIDE; ++z) {
            for (size_t y = 0; y < BLOCK_SIDE; ++y) {
                for (size_t x = 0; x < BLOCK_SIDE; ++x) {
                    size_t local = x + BLOCK_SIDE * y
                                    + BLOCK_SIDE * BLOCK_SIDE * z;

                    size_t t = bz * BLOCK_SIDE + z;
                    size_t lat = by * BLOCK_SIDE + y;
                    size_t lon = bx * BLOCK_SIDE + x;

                    size_t global = (t * SIDE + lat) * SIDE + lon;

                    float v = block[local];

                    if (memcmp(&v, &reconstructed[global],
                               sizeof(float)) != 0) {
                        fprintf(stderr,
                                "FAIL: block/chunk mismatch "
                                "b=%zu local=%zu global=%zu "
                                "block=%.9g chunk=%.9g\n",
                                b, local, global,
                                (double)v,
                                (double)reconstructed[global]);
                        return 0;
                    }

                    if (!isfinite(v) ||
                        (double)v < lower ||
                        (double)v > upper) {
                        fprintf(stderr,
                                "FAIL: metadata bound "
                                "b=%zu value=%.9g [%.9g, %.9g]\n",
                                b, (double)v, lower, upper);
                        return 0;
                    }

                    if ((double)v < actual_min)
                        actual_min = (double)v;

                    if ((double)v > actual_max)
                        actual_max = (double)v;

                    ++checked_values;
                }
            }
        }
    }

    if (checked_values != VALUES ||
        actual_min < (double)cmin ||
        actual_max > (double)cmax) {
        fprintf(stderr, "FAIL: chunk-level metadata\n");
        return 0;
    }

    printf("Physical blocks checked: %d\n", BLOCKS);
    printf("Reconstructed values checked: %zu\n", checked_values);
    printf("Reconstructed min / max (K): %.9g / %.9g\n",
           actual_min, actual_max);
    printf("Chunk metadata min / max (K): %.9g / %.9g\n",
           (double)cmin, (double)cmax);
    printf("Reserved metadata eps: %.9g\n", (double)eps);

    return 1;
}

int main(int argc, char **argv)
{
    const char *path = argc > 1
        ? argv[1]
        : "../data/era5_t2m_32cube_f32le.raw";

    if (argc > 2) {
        fprintf(stderr, "Usage: %s [path_to_RAW]\n", argv[0]);
        return 1;
    }

    const uint16_t endian = 1;
    if (*(const unsigned char *)&endian != 1u ||
        sizeof(float) != 4u) {
        fprintf(stderr, "FAIL: requires little-endian float32\n");
        return 1;
    }

    if (!read_raw(path))
        return 1;

    size_t packed_size = 0;

    SzResult status = szfp_pack_chunk(
        source,
        SIDE, SIDE, SIDE,
        8.0,
        BLOCK_SIDE,
        packed,
        sizeof(packed),
        &packed_size
    );

    if (status != SZ_OK || packed_size != PACKED_BYTES) {
        fprintf(stderr,
                "FAIL: pack code=%d size=%zu expected=%d\n",
                (int)status, packed_size, PACKED_BYTES);
        return 1;
    }

    if (!unpack_whole()) {
        fprintf(stderr, "FAIL: whole-chunk decode\n");
        return 1;
    }

    if (!check_blocks_and_metadata())
        return 1;

    double max_abs_error = 0.0;
    long double sum_squared_error = 0.0L;

    for (size_t i = 0; i < VALUES; ++i) {
        double error = fabs(
            (double)source[i] - (double)reconstructed[i]
        );

        if (error > max_abs_error)
            max_abs_error = error;

        sum_squared_error += (long double)error * error;
    }

    printf("\nInput: %s\n", path);
    printf("Shape: 32 x 32 x 32\n");
    printf("Axis order: time, latitude, longitude\n");
    printf("RAW bytes: %zu\n", sizeof(source));
    printf("ZFP payload bytes: %d\n", PAYLOAD_BYTES);
    printf("Metadata bytes: %d\n", METADATA_BYTES);
    printf("Total packed bytes: %zu\n", packed_size);
    printf("RAW / packed size: %.3fx\n",
           (double)sizeof(source) / (double)packed_size);
    printf("Original-to-stored max absolute error (K): %.9g\n",
           max_abs_error);
    printf("Original-to-stored RMSE (K): %.9Lg\n",
           sqrtl(sum_squared_error / VALUES));

    puts("\nALL ERA5 PACK CHECKS PASSED");
    return 0;
}