// DramSim stub：零延迟 DRAM，send_request 立即回调
// 替换 ramulator，用于功能验证阶段
#include "dram_sim.h"
#include "simx_log.h"          // 新增日志头文件

using namespace vortex;

class DramSim::Impl {
public:
    Impl(uint32_t num_channels, uint32_t /* channel_size */, float clock_ratio) {
        const uint32_t total_size = 64 * 1024 * 1024; 
        SIMX_LOG("\t[DRAM_STUB]: Impl created: channels=%u, total_size=%u, ratio=%.2f",
                 num_channels, total_size, clock_ratio);
    }

    void reset() {
        SIMX_LOG("\t[DRAM_STUB]: reset");
    }

    void tick() {
        // 每周期都被调用，为避免刷屏，不打印日志
    }

    void send_request(uint64_t addr, bool is_write,
                      ResponseCallback cb, void* arg) {
        SIMX_LOG("\t[DRAM_STUB]: send_request: addr=0x%lx, %s, cb=%p, arg=%p",
                 addr, is_write ? "write" : "read", (void*)cb, arg);
        if (cb) {
            SIMX_LOG("\t[DRAM_STUB]:   -> callback immediately");
            cb(arg);
        }
    }
};

DramSim::DramSim(uint32_t num_channels, uint32_t channel_size, float clock_ratio)
    : impl_(new Impl(num_channels, channel_size, clock_ratio)) {}

DramSim::~DramSim() { delete impl_; }

void DramSim::reset() { impl_->reset(); }

void DramSim::tick() { impl_->tick(); }

void DramSim::send_request(uint64_t addr, bool is_write,
                           ResponseCallback cb, void* arg) {
    impl_->send_request(addr, is_write, cb, arg);
}