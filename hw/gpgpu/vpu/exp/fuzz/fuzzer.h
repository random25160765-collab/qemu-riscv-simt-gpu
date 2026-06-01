/*
 * fuzzer.h — VPU Engine Fuzzer
 */
#ifndef VPU_FUZZER_H
#define VPU_FUZZER_H
#include <stdint.h>
#include "../state.h"
int fuzzer_run(GPGPUState *s, uint32_t seed, int rounds, int verbose);
int fuzzer_replay(GPGPUState *s, const char *hex_list, int verbose);
#endif
