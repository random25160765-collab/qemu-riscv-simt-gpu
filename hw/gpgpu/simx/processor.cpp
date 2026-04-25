// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "processor.h"
#include "processor_impl.h"
#include <cstdio>   // ← 新增

using namespace vortex;

ProcessorImpl::ProcessorImpl(const Arch& arch)
  : arch_(arch)
  , clusters_(arch.num_clusters())
{
  SimPlatform::instance().initialize();

  fprintf(stderr, "[SimX] ProcessorImpl: arch(num_threads=%d, num_warps=%d, num_cores=%d, num_clusters=%d)\n",
          arch.num_threads(), arch.num_warps(), arch.num_cores(), arch.num_clusters());

  // ... 中间代码不变 ...
}

int ProcessorImpl::run() {
  fprintf(stderr, "[SimX] ProcessorImpl::run() start\n");
  SimPlatform::instance().reset();
  this->reset();
  fprintf(stderr, "[SimX] ProcessorImpl::run() reset done, entering main loop\n");

  bool done;
  int exitcode = 0;
  int cycle_count = 0;
  do {
    SimPlatform::instance().tick();
    done = true;
    for (auto cluster : clusters_) {
      if (cluster->running()) {
        done = false;
        continue;
      }
      exitcode |= cluster->get_exitcode();
    }
    cycle_count++;
    if (cycle_count % 1000 == 0) {
      fprintf(stderr, "[SimX] run() cycle %d, still running...\n", cycle_count);
      if (cycle_count >= 10000) {
        fprintf(stderr, "[SimX] run() TIMEOUT after %d cycles, forcing exit\n", cycle_count);
        done = true;
        exitcode = -1;
      }
    }
    perf_mem_latency_ += perf_mem_pending_reads_;
  } while (!done);

  fprintf(stderr, "[SimX] ProcessorImpl::run() done, cycles=%d, exitcode=%d\n", cycle_count, exitcode);
  return exitcode;
}

void ProcessorImpl::reset() {
  fprintf(stderr, "[SimX] ProcessorImpl::reset()\n");
  perf_mem_reads_ = 0;
  perf_mem_writes_ = 0;
  perf_mem_latency_ = 0;
  perf_mem_pending_reads_ = 0;
}

bool ProcessorImpl::cycle() {
  if (!is_cycle_initialized_) {
    fprintf(stderr, "[SimX] ProcessorImpl: Initializing cycle()\n");
    SimPlatform::instance().reset();
    this->reset();
    is_cycle_initialized_ = true;
  }

  SimPlatform::instance().tick();
  bool anyRunning = false;
  for (auto cluster : clusters_) {
    if (cluster->running()) {
      anyRunning = true;
      break;
    }
  }
  perf_mem_latency_ += perf_mem_pending_reads_;
  return anyRunning;
}