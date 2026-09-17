#include <zfp.h>

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "core.h"
#include "util.h"

typedef struct {
    size_t nval;
    size_t bx;
    size_t by;
    size_t bz;
    size_t nblk;
    size_t data_size;
    size_t meta_size;
    size_t pack_size;
} SzfpLayout;

static int mul_size(size_t a, size_t b, size_t* out) {
    if (out == NULL) {
        return 0;
    }

    if (a != 0 && b > SIZE_MAX / a) {
        return 0;
    }

    *out = a * b;
    return 1;
}

static int add_size(size_t a, size_t b, size_t* out) {
    if (out == NULL) {
        return 0;
    }

    if (a > SIZE_MAX - b) {
        return 0;
    }

    *out = a + b;
    return 1;
}

static int calc_layout(
    size_t nx,
    size_t ny,
    size_t nz,
    double rate,
    int block_dim,
    SzfpLayout* lt
) {
    size_t nxy;
    size_t bxy;
    size_t off_size;
    size_t hdr_size;

    if (lt == NULL) {
        return 0;
    }

    if (nx == 0 || ny == 0 || nz == 0 || rate <= 0.0 || !isfinite(rate)) {
        return 0;
    }

    if (block_dim != 4) {
        return 0;
    }

    if (nx % (size_t)block_dim != 0 || ny % (size_t)block_dim != 0 ||
        nz % (size_t)block_dim != 0) {
        return 0;
    }

    memset(lt, 0, sizeof(*lt));

    if (!mul_size(nx, ny, &nxy) || !mul_size(nxy, nz, &lt->nval)) {
        return 0;
    }

    lt->bx = nx / (size_t)block_dim;
    lt->by = ny / (size_t)block_dim;
    lt->bz = nz / (size_t)block_dim;

    if (!mul_size(lt->bx, lt->by, &bxy) || !mul_size(bxy, lt->bz, &lt->nblk) ||
        !mul_size(lt->nblk, 2u, &off_size)) {
        return 0;
    }

    lt->data_size = (size_t)((double)lt->nval * rate / 8.0);
    hdr_size = 3u * sizeof(float);

    if (lt->data_size == 0 || !add_size(hdr_size, off_size, &lt->meta_size) ||
        !add_size(lt->data_size, lt->meta_size, &lt->pack_size)) {
        return 0;
    }

    return 1;
}

static SzResult decode_chunk(
    const unsigned char* src,
    size_t src_size,
    size_t nx,
    size_t ny,
    size_t nz,
    double rate,
    float* out
) {
    bitstream* stream;
    zfp_stream* zfp;
    zfp_field* field;
    size_t ret;

    stream = stream_open((void*)src, src_size);
    if (stream == NULL) {
        return SZ_ERR_STREAM;
    }

    zfp = zfp_stream_open(stream);
    if (zfp == NULL) {
        stream_close(stream);
        return SZ_ERR_ZFP;
    }

    zfp_stream_set_rate(zfp, rate, zfp_type_float, 3, 0);
    zfp_stream_rewind(zfp);

    field = zfp_field_3d(out, zfp_type_float, nx, ny, nz);
    if (field == NULL) {
        zfp_stream_close(zfp);
        stream_close(stream);
        return SZ_ERR_FIELD;
    }

    ret = zfp_decompress(zfp, field);

    zfp_field_free(field);
    zfp_stream_close(zfp);
    stream_close(stream);

    return ret == 0 ? SZ_ERR_DECOMPRESS : SZ_OK;
}

SzResult szfp_decompress_block(
    const unsigned char* block_bytes,
    size_t block_nbytes,
    double rate,
    int zfp_type_value,
    int dims,
    void* out
) {
    bitstream* stream;
    zfp_stream* zfp;
    zfp_field* field;
    size_t ret;
    zfp_type type = (zfp_type)zfp_type_value;

    if (block_bytes == NULL || out == NULL) {
        return SZ_ERR_NULL;
    }

    if (block_nbytes == 0 || rate <= 0.0 || !isfinite(rate)) {
        return SZ_ERR_ARG;
    }

    stream = stream_open((void*)block_bytes, block_nbytes);
    if (stream == NULL) {
        return SZ_ERR_STREAM;
    }

    zfp = zfp_stream_open(stream);
    if (zfp == NULL) {
        stream_close(stream);
        return SZ_ERR_ZFP;
    }

    zfp_stream_set_rate(zfp, rate, type, dims, 0);
    zfp_stream_rewind(zfp);

    field = create_block_field(out, type, dims);
    if (field == NULL) {
        zfp_stream_close(zfp);
        stream_close(stream);
        return SZ_ERR_DIMS;
    }

    ret = zfp_decompress(zfp, field);

    zfp_field_free(field);
    zfp_stream_close(zfp);
    stream_close(stream);

    return ret == 0 ? SZ_ERR_DECOMPRESS : SZ_OK;
}

SzResult szfp_pack_chunk(
    const float* chunk,
    size_t nx,
    size_t ny,
    size_t nz,
    double rate,
    int block_dim,
    unsigned char* out,
    size_t out_capacity,
    size_t* out_size
) {
    SzfpLayout lt;
    bitstream* stream;
    zfp_stream* zfp;
    zfp_field* field;
    float* block_mins;
    float* block_maxs;
    float cmin;
    float cmax;
    float eps;
    unsigned char* meta;
    unsigned char* offs;
    size_t ret;
    size_t block_nbytes;
    size_t minmax_bytes;

    if (chunk == NULL || out == NULL || out_size == NULL) {
        return SZ_ERR_NULL;
    }

    if (!calc_layout(nx, ny, nz, rate, block_dim, &lt)) {
        return SZ_ERR_ARG;
    }

    if (out_capacity < lt.pack_size) {
        return SZ_ERR_SIZE;
    }

    stream = stream_open(out, lt.data_size);
    if (stream == NULL) {
        return SZ_ERR_STREAM;
    }

    zfp = zfp_stream_open(stream);
    if (zfp == NULL) {
        stream_close(stream);
        return SZ_ERR_ZFP;
    }

    zfp_stream_set_rate(zfp, rate, zfp_type_float, 3, 0);
    zfp_stream_rewind(zfp);

    field = zfp_field_3d((void*)chunk, zfp_type_float, nx, ny, nz);
    if (field == NULL) {
        zfp_stream_close(zfp);
        stream_close(stream);
        return SZ_ERR_FIELD;
    }

    ret = zfp_compress(zfp, field);

    zfp_field_free(field);
    zfp_stream_close(zfp);
    stream_close(stream);

    if (ret == 0) {
        return SZ_ERR_COMPRESS;
    }

    if (lt.nblk == 0 || lt.data_size % lt.nblk != 0) {
        return SZ_ERR_SIZE;
    }

    block_nbytes = lt.data_size / lt.nblk;

    if (block_nbytes == 0 || (double)block_nbytes != 64.0 * rate / 8.0) {
        return SZ_ERR_ARG;
    }

    if (!mul_size(lt.nblk, sizeof(float), &minmax_bytes)) {
        return SZ_ERR_SIZE;
    }

    block_mins = (float*)malloc(minmax_bytes);
    block_maxs = (float*)malloc(minmax_bytes);

    if (block_mins == NULL || block_maxs == NULL) {
        free(block_mins);
        free(block_maxs);
        return SZ_ERR_MALLOC;
    }

    cmin = FLT_MAX;
    cmax = -FLT_MAX;
    eps = 0.0f;

    for (size_t block_id = 0; block_id < lt.nblk; block_id++) {
        float block[64];
        float bmin = FLT_MAX;
        float bmax = -FLT_MAX;

        SzResult code = szfp_decompress_block(
            out + block_id * block_nbytes, block_nbytes, rate, zfp_type_float, 3, block
        );

        if (code != SZ_OK) {
            free(block_mins);
            free(block_maxs);
            return code;
        }

        for (size_t i = 0; i < 64; i++) {
            float v = block[i];

            if (v < bmin) {
                bmin = v;
            }

            if (v > bmax) {
                bmax = v;
            }
        }

        block_mins[block_id] = bmin;
        block_maxs[block_id] = bmax;

        if (bmin < cmin) {
            cmin = bmin;
        }

        if (bmax > cmax) {
            cmax = bmax;
        }
    }

    meta = out + lt.data_size;
    memcpy(meta, &cmin, sizeof(float));
    memcpy(meta + sizeof(float), &cmax, sizeof(float));
    memcpy(meta + 2u * sizeof(float), &eps, sizeof(float));

    offs = meta + 3u * sizeof(float);

    if (cmax == cmin) {
        memset(offs, 0, lt.nblk * 2u);
    } else {
        double span = (double)cmax - (double)cmin;
        double scale = 255.0 / span;

        for (size_t block_id = 0; block_id < lt.nblk; block_id++) {
            int min_off =
                (int)floor(((double)block_mins[block_id] - (double)cmin) * scale);

            int max_off =
                (int)ceil(((double)block_maxs[block_id] - (double)cmin) * scale);

            offs[block_id * 2u] = to_u8(min_off);
            offs[block_id * 2u + 1u] = to_u8(max_off);
        }
    }
    free(block_mins);
    free(block_maxs);
    *out_size = lt.pack_size;
    return SZ_OK;
}