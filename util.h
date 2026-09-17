#ifndef SKIPZFP_UTIL_H
#define SKIPZFP_UTIL_H

#include <stddef.h>
#include <zfp.h>

size_t get_index3(size_t x, size_t y, size_t z, size_t ny, size_t nz);

unsigned char to_u8(int x);

zfp_field* create_block_field(void* out, zfp_type type, int dims);

#endif