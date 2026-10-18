#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <zfp.h>

/* Decode one zfpy-compressed chunk (a full ZFP header, then the stream) into dst,
 * reading straight from buf, so no copy is made and no Python runs while it
 * decodes. raw_nbytes is the size the decoded chunk must have. */
int zc_decode(
    const unsigned char* buf,
    size_t buf_size,
    size_t raw_nbytes,
    float* dst
) {
    int code = 0;
    bitstream* s = stream_open((void*)buf, buf_size);
    zfp_stream* z = zfp_stream_open(s);
    zfp_field* f = zfp_field_alloc();

    if (s == NULL || z == NULL || f == NULL) {
        code = 4;
    } else if (!zfp_read_header(z, f, ZFP_HEADER_FULL)) {
        code = 1;
    } else if (f->type != zfp_type_float || zfp_field_size(f, NULL) * 4 != raw_nbytes) {
        code = 2;
    } else {
        zfp_field_set_pointer(f, dst);
        if (!zfp_decompress(z, f)) {
            code = 3;
        }
    }
    if (f != NULL) {
        zfp_field_free(f);
    }
    if (z != NULL) {
        zfp_stream_close(z);
    }
    if (s != NULL) {
        stream_close(s);
    }
    return code;
}

/* Decode one chunk as zc_decode() does and count the values above th. */
int zc_count_gt(
    const unsigned char* buf,
    size_t buf_size,
    size_t raw_nbytes,
    double th,
    size_t* out
) {
    float* tmp = (float*)malloc(raw_nbytes);
    size_t c = 0;
    int code;

    if (tmp == NULL) {
        return 4;
    }
    code = zc_decode(buf, buf_size, raw_nbytes, tmp);
    if (code == 0) {
        for (size_t i = 0; i < raw_nbytes / 4; i++) {
            c += (double)tmp[i] > th;
        }
        *out = c;
    }
    free(tmp);
    return code;
}
