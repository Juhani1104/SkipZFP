
#define main era5_batch_embedded_main
#include "test_era5_batch.c"
#undef main

int main(void)
{
    const char *packed_path = "../data/era5_128_packed.bin";
    const char *eps_path = "../data/era5_128_prefix_eps.bin";

    read_raw("../data/era5_t2m_128cube_f32le.raw");
    build_store();

    FILE *packed_file = fopen(packed_path, "wbx");
    if (!packed_file) {
        perror(packed_path);
        return 1;
    }

    if (fwrite(store, 1, sizeof(store), packed_file) != sizeof(store) ||
        fclose(packed_file) != 0) {
        fprintf(stderr, "FAIL: packed export\n");
        return 1;
    }

    FILE *eps_file = fopen(eps_path, "wbx");
    if (!eps_file) {
        perror(eps_path);
        return 1;
    }

    for (size_t ch = 0; ch < CHUNKS; ++ch) {
        if (fwrite(error_bound[ch], sizeof(double), 3, eps_file) != 3) {
            fprintf(stderr, "FAIL: eps export\n");
            fclose(eps_file);
            return 1;
        }
    }

    if (fclose(eps_file) != 0) {
        fprintf(stderr, "FAIL: eps close\n");
        return 1;
    }

    printf("Packed file: %s (%zu bytes)\n",
           packed_path, sizeof(store));
    printf("Prefix bounds: %s (%zu bytes)\n",
           eps_path, (size_t)CHUNKS * 3 * sizeof(double));
    puts("ERA5 HTTP EXPORT COMPLETE");
    return 0;
}