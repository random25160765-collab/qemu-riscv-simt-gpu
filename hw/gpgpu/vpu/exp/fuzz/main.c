/*
 * fuzz/main.c — VPU Fuzzer Entry Point
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "state.h"
#include "config.h"
#include "fuzzer.h"

int main(int argc, char **argv)
{
    GPGPUState s;
    memset(&s, 0, sizeof(s));
    vpu_config_default(&s.cfg);
    s.vram_size = s.cfg.vram_mb * 1024 * 1024;
    s.global_status = 1;
    s.vram_ptr = calloc(1, s.vram_size);
    if (!s.vram_ptr) {
        fprintf(stderr, "VRAM alloc failed\n");
        return 1;
    }

    uint32_t seed = 0xFFFFFFFF;
    int rounds = 10000;
    int verbose = 1;
    char *replay_raw = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-s") && i + 1 < argc)
            seed = (uint32_t)strtoul(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "-n") && i + 1 < argc)
            rounds = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-q"))
            verbose = 0;
        else if (!strcmp(argv[i], "-d"))
            verbose = 2;
        else if (!strcmp(argv[i], "-r") && i + 1 < argc)
            replay_raw = argv[++i];
    }

    int errors;
    if (replay_raw)
        errors = fuzzer_replay(&s, replay_raw, verbose);
    else
        errors = fuzzer_run(&s, seed, rounds, verbose);
    free(s.vram_ptr);
    return errors ? 1 : 0;
}
