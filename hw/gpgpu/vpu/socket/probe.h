#ifndef VPU_PROBE_H
#define VPU_PROBE_H

#include "ring/ring.h"

typedef struct ProbeConfig {
    const char *slow_path;
    const char *fast_path;
    ring_buf   *slow_ring;
    ring_buf   *fast_ring;
} ProbeConfig;

int  probe_init(ProbeConfig *cfg);
void probe_start(void);
void probe_stop(void);

#endif
