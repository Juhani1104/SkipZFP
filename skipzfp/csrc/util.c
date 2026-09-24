#include "util.h"

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