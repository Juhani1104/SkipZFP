#include "util.h"

#include <math.h>
#include <string.h>

size_t get_index3(size_t x, size_t y, size_t z, size_t ny, size_t nz) {
    return x * ny * nz + y * nz + z;
}

unsigned char to_u8(int x) {
    if (x < 0) {
        return 0;
    }

    if (x > 255) {
        return 255;
    }

    return (unsigned char)x;
}

zfp_field* create_block_field(void* out, zfp_type type, int dims) {
    switch (dims) {
    case 1:
        return zfp_field_1d(out, type, 4);
    case 2:
        return zfp_field_2d(out, type, 4, 4);
    case 3:
        return zfp_field_3d(out, type, 4, 4, 4);
    case 4:
        return zfp_field_4d(out, type, 4, 4, 4, 4);
    default:
        return NULL;
    }
}

int szfp_calc_layout(
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

    if (lt->data_size == 0 || !add_size(hdr_size, off_size, &lt->meta_size)) {
        return 0;
    }

    return 1;
}
