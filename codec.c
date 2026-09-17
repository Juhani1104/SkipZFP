#include <zfp.h>

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "core.h"


static int mul_size(size_t a, size_t b, size_t* out)
{
    if (out == NULL) {
        return 0;
    }

    if (a != 0 && b > SIZE_MAX / a) {
        return 0;
    }

    *out = a * b;
    return 1;
}


static int add_size(size_t a, size_t b, size_t* out)
{
    if (out == NULL) {
        return 0;
    }

    if (a > SIZE_MAX - b) {
        return 0;
    }

    *out = a + b;
    return 1;
}


static int calc_size(
    size_t nx,
    size_t ny,
    size_t nz,
    double rate,
    int block_dim,
    size_t* data_size,
    size_t* meta_size,
    size_t* pack_size
) {
    size_t nxy;
    size_t nval;
    size_t bx;
    size_t by;
    size_t bz;
    size_t bxy;
    size_t nblk;
    size_t off_size;
    size_t hdr_size;

    if (data_size == NULL || meta_size == NULL || pack_size == NULL) {
        return 0;
    }

    if (nx == 0 || ny == 0 || nz == 0 || rate <= 0.0 || !isfinite(rate)) {
        return 0;
    }

    if (block_dim <= 0) {
        return 0;
    }

    if (
        nx % (size_t)block_dim != 0 ||
        ny % (size_t)block_dim != 0 ||
        nz % (size_t)block_dim != 0
    ) {
        return 0;
    }

    if (
        !mul_size(nx, ny, &nxy) ||
        !mul_size(nxy, nz, &nval)
    ) {
        return 0;
    }

    bx = nx / (size_t)block_dim;
    by = ny / (size_t)block_dim;
    bz = nz / (size_t)block_dim;

    if (
        !mul_size(bx, by, &bxy) ||
        !mul_size(bxy, bz, &nblk) ||
        !mul_size(nblk, 2u, &off_size)
    ) {
        return 0;
    }

    *data_size = (size_t)((double)nval * rate / 8.0);
    hdr_size = 3u * sizeof(float);

    if (
        *data_size == 0 ||
        !add_size(hdr_size, off_size, meta_size) ||
        !add_size(*data_size, *meta_size, pack_size)
    ) {
        return 0;
    }

    return 1;
}


SzResult szfp_packed_size(
    size_t nx,
    size_t ny,
    size_t nz,
    double rate,
    int block_dim,
    size_t* out_size
) {
    size_t data_size;
    size_t meta_size;
    size_t pack_size;

    if (out_size == NULL) {
        return SZ_ERR_NULL;
    }

    if (
        !calc_size(
            nx,
            ny,
            nz,
            rate,
            block_dim,
            &data_size,
            &meta_size,
            &pack_size
        )
    ) {
        return SZ_ERR_ARG;
    }

    *out_size = pack_size;
    return SZ_OK;
}


SzResult szfp_unpack_chunk(
    const unsigned char* packed,
    size_t packed_size,
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
    size_t meta_size;
    size_t pack_size;
    bitstream* stream;
    zfp_stream* zfp;
    zfp_field* field;
    size_t ret;

    if (packed == NULL || out == NULL) {
        return SZ_ERR_NULL;
    }

    if (
        !mul_size(nx, ny, &nxy) ||
        !mul_size(nxy, nz, &nval)
    ) {
        return SZ_ERR_SIZE;
    }

    if (out_count < nval) {
        return SZ_ERR_SIZE;
    }

    if (
        !calc_size(
            nx,
            ny,
            nz,
            rate,
            block_dim,
            &data_size,
            &meta_size,
            &pack_size
        )
    ) {
        return SZ_ERR_ARG;
    }

    if (packed_size != pack_size) {
        return SZ_ERR_SIZE;
    }

    stream = stream_open((void*)packed, data_size);
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


SzResult szfp_layout_size(
    size_t nx,
    size_t ny,
    size_t nz,
    double rate,
    int block_dim,
    size_t* data_size,
    size_t* meta_size,
    size_t* pack_size
) {
    if (data_size == NULL || meta_size == NULL || pack_size == NULL) {
        return SZ_ERR_NULL;
    }

    if (
        !calc_size(
            nx,
            ny,
            nz,
            rate,
            block_dim,
            data_size,
            meta_size,
            pack_size
        )
    ) {
        return SZ_ERR_ARG;
    }

    return SZ_OK;
}