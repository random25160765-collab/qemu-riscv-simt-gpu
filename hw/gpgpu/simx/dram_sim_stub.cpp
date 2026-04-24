// DramSim stub：零延迟 DRAM，send_request 立即回调
// 替换 ramulator，用于功能验证阶段
#include "dram_sim.h"

using namespace vortex;

class DramSim::Impl {
public:
    Impl(uint32_t, uint32_t, float) {}
    void reset() {}
    void tick() {}
    void send_request(uint64_t /*addr*/, bool /*is_write*/,
                      ResponseCallback cb, void* arg) {
        if (cb) cb(arg);
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
