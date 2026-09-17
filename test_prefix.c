
#include <zfp.h>

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "core.h"



enum {
    SIDE = 8,
    VALUES = SIDE * SIDE * SIDE,
    BLOCK_VALUES = 64,
    BLOCKS = VALUES / BLOCK_VALUES,
    FULL_BLOCK_BYTES = 64,
    BUFFER_BYTES = 1024
};

static int encode_reference(
    const float input[VALUES],
    double rate,
    unsigned char output[BUFFER_BYTES],
    size_t *output_size
) {
    bitstream *bs = NULL;
    zfp_stream *zfp = NULL;
    zfp_field *field = NULL;
    double actual_rate;
    size_t written = 0;

    memset(output, 0, BUFFER_BYTES);
    bs = stream_open(output, BUFFER_BYTES);
    if (bs == NULL)
        goto cleanup;

    zfp = zfp_stream_open(bs);
    if (zfp == NULL)
        goto cleanup;

    actual_rate = zfp_stream_set_rate(
        zfp, rate, zfp_type_float, 3, 0
    );
    if (fabs(actual_rate - rate) > 1e-9) {
        fprintf(stderr,
                "Unsupported rate: requested %.1f, actual %.6f\n",
                rate, actual_rate);
        goto cleanup;
    }

    field = zfp_field_3d(
        (void *)input, zfp_type_float, SIDE, SIDE, SIDE
    );
    if (field == NULL)
        goto cleanup;

    zfp_stream_rewind(zfp);
    written = zfp_compress(zfp, field);

cleanup:
    if (field != NULL)
        zfp_field_free(field);
    if (zfp != NULL)
        zfp_stream_close(zfp);
    if (bs != NULL)
        stream_close(bs);

    if (written == 0) {
        fprintf(stderr, "Independent encoding failed at rate %.1f\n", rate);
        return 0;
    }

    *output_size = written;
    return 1;
}

static int decode_reference(
    const unsigned char *payload,
    size_t payload_size,
    double rate,
    float output[VALUES]
) {
    bitstream *bs = NULL;
    zfp_stream *zfp = NULL;
    zfp_field *field = NULL;
    size_t decoded = 0;

    bs = stream_open((void *)payload, payload_size);
    if (bs == NULL)
        goto cleanup;

    zfp = zfp_stream_open(bs);
    if (zfp == NULL)
        goto cleanup;

    if (fabs(zfp_stream_set_rate(
            zfp, rate, zfp_type_float, 3, 0
        ) - rate) > 1e-9)
        goto cleanup;

    field = zfp_field_3d(
        output, zfp_type_float, SIDE, SIDE, SIDE
    );
    if (field == NULL)
        goto cleanup;

    zfp_stream_rewind(zfp);
    decoded = zfp_decompress(zfp, field);

cleanup:
    if (field != NULL)
        zfp_field_free(field);
    if (zfp != NULL)
        zfp_stream_close(zfp);
    if (bs != NULL)
        stream_close(bs);

    if (decoded == 0) {
        fprintf(stderr, "Independent decoding failed at rate %.1f\n", rate);
        return 0;
    }

    return 1;
}

static int decode_only_prefix(
    const unsigned char *block,
    size_t prefix_bytes,
    double rate,
    float output[BLOCK_VALUES]
) {

    unsigned char padded[FULL_BLOCK_BYTES] = {0};
    bitstream *bs = NULL;
    zfp_stream *zfp = NULL;
    size_t consumed_bits = 0;

    if (prefix_bytes > FULL_BLOCK_BYTES)
        return 0;

    memcpy(padded, block, prefix_bytes);

    bs = stream_open(padded, sizeof(padded));
    if (bs == NULL)
        goto cleanup;

    zfp = zfp_stream_open(bs);
    if (zfp == NULL)
        goto cleanup;

    if (fabs(zfp_stream_set_rate(
            zfp, rate, zfp_type_float, 3, 0
        ) - rate) > 1e-9)
        goto cleanup;

    zfp_stream_rewind(zfp);

    /* IMPORTANT: low-level block API, NOT zfp_decompress(). */
    consumed_bits = zfp_decode_block_float_3(zfp, output);

cleanup:
    if (zfp != NULL)
        zfp_stream_close(zfp);
    if (bs != NULL)
        stream_close(bs);

    return consumed_bits != 0 &&
           consumed_bits <= prefix_bytes * 8u;
}


static void extract_reference_block(
    const float decoded_chunk[VALUES],
    size_t block_id,
    float block[BLOCK_VALUES]
) {
    size_t bx = block_id % 2u;
    size_t by = (block_id / 2u) % 2u;
    size_t bz = block_id / 4u;

    for (size_t i = 0; i < BLOCK_VALUES; ++i) {
        size_t x = bx * 4u + i % 4u;
        size_t y = by * 4u + (i / 4u) % 4u;
        size_t z = bz * 4u + i / 16u;
        block[i] = decoded_chunk[x + SIDE * (y + SIDE * z)];
    }
}

static void make_sample(float data[VALUES], int sample) {
    for (size_t i = 0; i < VALUES; ++i) {
        float x = (float)(i % SIDE);
        float y = (float)((i / SIDE) % SIDE);
        float z = (float)(i / (SIDE * SIDE));

        switch (sample) {
            case 0: 
                data[i] = 280.0f + 0.31f * x + 0.17f * y +
                          0.09f * z + 0.03f * sinf(x + y);
                break;
            case 1: 
                data[i] = 5.0f * sinf(0.6f * x + 0.2f * z) -
                          4.0f * cosf(0.5f * y) + 0.2f * z;
                break;
            case 2: 
                data[i] = 293.15f;
                break;
            default: 
                data[i] = (float)((i * 73u + 19u) % 211u) * 0.19f - 17.0f;
                break;
        }
    }
}

int main(void) {
    const double rates[] = {2.0, 4.0, 6.0, 8.0};
    const char *samples[] = {
        "temperature", "wind", "constant", "irregular"
    };

    float input[VALUES];
    float reference_decoded[VALUES];
    float prefix_decoded[BLOCK_VALUES];
    float reference_block[BLOCK_VALUES];

    unsigned char skipzfp_packed[BUFFER_BYTES];
    unsigned char independent_packed[BUFFER_BYTES];

    size_t skipzfp_size = 0;
    size_t independent_size = 0;
    int checked_blocks = 0;

    puts("SkipZFP prefix feasibility test");
    puts("Ground truth for each rate: independently encoded ZFP data.");
    puts("");

    for (int sample = 0; sample < 4; ++sample) {
        make_sample(input, sample);

        SzResult result = szfp_pack_chunk(
            input,
            SIDE, SIDE, SIDE,
            8.0, 4,
            skipzfp_packed,
            sizeof(skipzfp_packed),
            &skipzfp_size
        );

        if (result != SZ_OK || skipzfp_size != 512u + 12u + 16u) {
            fprintf(stderr,
                    "FAIL %s: SkipZFP packing, code=%d size=%zu\n",
                    samples[sample], (int)result, skipzfp_size);
            return 1;
        }

        for (size_t level = 0; level < 4; ++level) {
            double rate = rates[level];
            size_t prefix_bytes = (size_t)(BLOCK_VALUES * rate / 8.0);
            size_t expected_size = BLOCKS * prefix_bytes;

            if (!encode_reference(
                    input, rate,
                    independent_packed, &independent_size
                ))
                return 1;

            if (independent_size != expected_size) {
                fprintf(stderr,
                        "FAIL %s @ %.0f bits/value: "
                        "expected %zu payload bytes, got %zu\n",
                        samples[sample], rate,
                        expected_size, independent_size);
                return 1;
            }

            if (!decode_reference(
                    independent_packed,
                    independent_size,
                    rate,
                    reference_decoded
                ))
                return 1;

            for (size_t b = 0; b < BLOCKS; ++b) {
                const unsigned char *full_block =
                    skipzfp_packed + b * FULL_BLOCK_BYTES;
                const unsigned char *short_block =
                    independent_packed + b * prefix_bytes;

                if (memcmp(
                        full_block, short_block, prefix_bytes
                    ) != 0) {
                    fprintf(stderr,
                            "FAIL %s @ %.0f bits/value block %zu: "
                            "prefix bytes differ from independent encoding\n",
                            samples[sample], rate, b);
                    return 1;
                }

                if (!decode_only_prefix(
                        full_block,
                        prefix_bytes,
                        rate,
                        prefix_decoded
                    )) {
                    fprintf(stderr,
                            "FAIL %s @ %.0f bits/value block %zu: "
                            "low-level prefix decoding failed\n",
                            samples[sample], rate, b);
                    return 1;
                }

                extract_reference_block(
                    reference_decoded, b, reference_block
                );

                if (memcmp(
                        prefix_decoded,
                        reference_block,
                        sizeof(prefix_decoded)
                    ) != 0) {
                    fprintf(stderr,
                            "FAIL %s @ %.0f bits/value block %zu: "
                            "decoded float32 values differ\n",
                            samples[sample], rate, b);
                    return 1;
                }

                ++checked_blocks;
            }

            printf(
                "PASS %-12s rate=%g  prefix=%2zu B/block  blocks=%d\n",
                samples[sample], rate, prefix_bytes, BLOCKS
            );
        }
    }

    printf("\nALL PREFIX TESTS PASSED (%d block-level checks)\n",
           checked_blocks);
    return 0;
}