#include <zfp.h>

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "core.h"
#include "util.h"

SzResult szfp_decode(
    const unsigned char* src,
    size_t src_size,
    size_t nx,
    size_t ny,
    size_t nz,
    double rate,
    int block_dim,
    float* out,
    size_t out_count
) {
    size_t nxy;
    size_t nval;
    size_t data_size;
    SzfpLayout lt;
    bitstream* stream;
    zfp_stream* zfp;
    zfp_field* field;
    size_t ret;

    if (src == NULL || out == NULL) {
        return SZ_ERR_NULL;
    }

    if (!mul_size(nx, ny, &nxy) || !mul_size(nxy, nz, &nval)) {
        return SZ_ERR_SIZE;
    }

    if (out_count < nval) {
        return SZ_ERR_SIZE;
    }

    if (!szfp_calc_layout(nx, ny, nz, rate, block_dim, &lt)) {
        return SZ_ERR_ARG;
    }
    data_size = lt.data_size;

    if (src_size != data_size) {
        return SZ_ERR_SIZE;
    }

    stream = stream_open((void*)src, data_size);
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

    field = zfp_field_3d(out, zfp_type_float, nz, ny, nx);
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

SzResult szfp_layout(
    size_t nx,
    size_t ny,
    size_t nz,
    double rate,
    int block_dim,
    size_t* data_size,
    size_t* meta_size
) {
    SzfpLayout lt;

    if (data_size == NULL || meta_size == NULL) {
        return SZ_ERR_NULL;
    }

    if (!szfp_calc_layout(nx, ny, nz, rate, block_dim, &lt)) {
        return SZ_ERR_ARG;
    }

    *data_size = lt.data_size;
    *meta_size = lt.meta_size;

    return SZ_OK;
}