#define _POSIX_C_SOURCE 199309L

#include "conflux_platform.h"
#include "conflux_device.h"
#include "conflux_allocator.h"
#include "conflux_queue.h"
#include "conflux_kernel.h"
#include "conflux_cmd_builder.h"
#include "conflux_event.h"
#include "conflux_log.h"
#include "conflux_error.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>

/* ================================================================
 * 模拟设备执行回调
 * ================================================================ */
int simx_execute(conflux_cmd_t *cmd, void *user_data) 
{
    (void)user_data;

    conflux_perf_record_submit(cmd->type);
    
    switch (cmd->type) {
        case CONFLUX_CMD_COPY:
            CONFLUX_DEBUG("SimX: copy 0x%lx → 0x%lx (%u bytes)",
                         (unsigned long)cmd->src_addr,
                         (unsigned long)cmd->dst_addr,
                         cmd->size);
            conflux_perf_record_copy(cmd->size);
            usleep(10000);  /* 模拟 10ms DMA */
            break;
            
        case CONFLUX_CMD_KERNEL:
            CONFLUX_DEBUG("SimX: kernel %u at 0x%lx, %u work-items",
                         cmd->kernel_id,
                         (unsigned long)cmd->src_addr,
                         cmd->size);
            conflux_perf_record_submit(cmd->type);
            usleep(50000);  /* 模拟 50ms 内核执行 */
            break;
            
        case CONFLUX_CMD_ALLOC:
            CONFLUX_DEBUG("SimX: alloc 0x%lx (%u bytes)",
                         (unsigned long)cmd->src_addr, cmd->size);
            conflux_perf_record_alloc(cmd->size);
            break;
            
        case CONFLUX_CMD_FREE:
            CONFLUX_DEBUG("SimX: free 0x%lx", (unsigned long)cmd->src_addr);
            break;
            
        case CONFLUX_CMD_BARRIER:
            CONFLUX_DEBUG("SimX: barrier");
            break;
            
        default:
            break;
    }
    
    return 0;
}

/* ================================================================
 * 主测试
 * ================================================================ */
int main(void) 
{
    int ret;
    
    /* ---- 初始化日志和性能统计 ---- */
    conflux_log_init(CONFLUX_LOG_INFO);
    conflux_perf_init();
    
    CONFLUX_INFO("=== Conflux Runtime Integration Test ===");
    
    /* ============================================================
     * 阶段 1：平台初始化 + 设备发现
     * ============================================================ */
    CONFLUX_INFO("--- Phase 1: Platform Init ---");
    
    ret = conflux_platform_init();
    assert(ret == CONFLUX_SUCCESS);
    
    /* 模拟 3 个 SimX 后端 */
    conflux_device_desc_t descs[3];
    
    /* GPU 0：64MB 小 GPU（SIM 模式：空 path） */
    memset(&descs[0], 0, sizeof(descs[0]));
    descs[0].path[0]   = '\0';
    descs[0].mmio_base = 0x10000000;
    descs[0].mmio_size = 64 * 1024 * 1024;
    descs[0].vendor_id = 0x1AF4;
    descs[0].device_id = 0x1050;
    descs[0].available = 1;
    
    /* GPU 1：128MB 中 GPU（SIM 模式） */
    memset(&descs[1], 0, sizeof(descs[1]));
    descs[1].path[0]   = '\0';
    descs[1].mmio_base = 0x20000000;
    descs[1].mmio_size = 128 * 1024 * 1024;
    descs[1].vendor_id = 0x1AF4;
    descs[1].device_id = 0x1051;
    descs[1].available = 1;
    
    /* GPU 2：256MB 大 GPU（SIM 模式） */
    memset(&descs[2], 0, sizeof(descs[2]));
    descs[2].path[0]   = '\0';
    descs[2].mmio_base = 0x30000000;
    descs[2].mmio_size = 256 * 1024 * 1024;
    descs[2].vendor_id = 0x1AF4;
    descs[2].device_id = 0x1052;
    descs[2].available = 1;
    
    for (int i = 0; i < 3; i++) {
        ret = conflux_platform_probe_specific(&descs[i]);
        assert(ret == i);
    }
    assert(conflux_platform_get_num_devices() == 3);
    conflux_platform_dump();
    
    /* 选择 GPU 1（128MB）作为测试设备 */
    ret = conflux_platform_open_device(1);
    assert(ret == CONFLUX_SUCCESS);
    
    conflux_device_t *dev = conflux_platform_get_device(1);
    assert(dev != NULL);
    assert(conflux_device_is_online(dev));
    
    CONFLUX_INFO("Selected device 1: %s", dev->name);
    
    /* ============================================================
     * 阶段 2：创建命令队列
     * ============================================================ */
    CONFLUX_INFO("--- Phase 2: Create Command Queue ---");
    
    conflux_queue_t *queue = conflux_queue_create(16, simx_execute, NULL);
    assert(queue != NULL);
    conflux_queue_start_consumer(queue);
    
    CONFLUX_INFO("Command queue created (16 entries)");
    
    /* ============================================================
     * 阶段 3：创建命令构建器
     * ============================================================ */
    CONFLUX_INFO("--- Phase 3: Create Command Builder ---");
    
    conflux_allocator_t *alloc = conflux_device_get_allocator(dev);
    assert(alloc != NULL);
    
    conflux_cmd_builder_t *builder = conflux_cmd_builder_create(queue, alloc);
    assert(builder != NULL);
    
    /* ============================================================
     * 阶段 4：分配缓冲区（对应 clCreateBuffer）
     * ============================================================ */
    CONFLUX_INFO("--- Phase 4: Allocate Buffers ---");
    
    uint64_t buf_a, buf_b, buf_c;
    
    ret = conflux_cmd_builder_alloc(builder, 4096, &buf_a, NULL);
    assert(ret == CONFLUX_SUCCESS);
    CONFLUX_INFO("  buf_a = 0x%lx", (unsigned long)buf_a);
    
    ret = conflux_cmd_builder_alloc(builder, 4096, &buf_b, NULL);
    assert(ret == CONFLUX_SUCCESS);
    CONFLUX_INFO("  buf_b = 0x%lx", (unsigned long)buf_b);
    
    ret = conflux_cmd_builder_alloc(builder, 4096, &buf_c, NULL);
    assert(ret == CONFLUX_SUCCESS);
    CONFLUX_INFO("  buf_c = 0x%lx", (unsigned long)buf_c);
    
    /* 验证地址不重叠 */
    assert(buf_a != buf_b);
    assert(buf_b != buf_c);
    assert(buf_a != buf_c);
    
    conflux_allocator_dump(alloc);
    
    /* ============================================================
     * 阶段 5：数据传输（对应 clEnqueueWriteBuffer）
     * ============================================================ */
    CONFLUX_INFO("--- Phase 5: Write Data to Device ---");
    
    conflux_copy_request_t write_req = {
        .src_addr = 0,      /* 特殊值：表示主机内存 */
        .dst_addr = buf_a,
        .size     = 4096,
    };
    
    conflux_event_t *write_ev = NULL;
    ret = conflux_cmd_builder_copy(builder, &write_req, &write_ev);
    assert(ret == CONFLUX_SUCCESS);
    
    uint64_t t0 = conflux_get_time_ns();
    conflux_event_wait(write_ev, 0);
    uint64_t t1 = conflux_get_time_ns();
    conflux_perf_record_wait(t1 - t0);
    
    CONFLUX_INFO("  buf_a written, waited %lu ns", (unsigned long)(t1 - t0));
    conflux_event_destroy(write_ev);
    
    /* 写第二块 */
    write_req.dst_addr = buf_b;
    ret = conflux_cmd_builder_copy(builder, &write_req, &write_ev);
    assert(ret == CONFLUX_SUCCESS);
    conflux_event_wait(write_ev, 0);
    conflux_event_destroy(write_ev);
    
    /* ============================================================
     * 阶段 6：编译并上传内核（对应 clBuildProgram）
     * ============================================================ */
    CONFLUX_INFO("--- Phase 6: Upload Kernel ---");
    
    /* 模拟 Conflux RISC-V 指令 */
    uint8_t vec_add_instr[] = {
        0x13, 0x01, 0x01, 0xFF,   /* addi sp, sp, -16 */
        0x83, 0x25, 0x00, 0x00,   /* lw a1, 0(x0)  — buf_a */
        0x03, 0x26, 0x00, 0x00,   /* lw a2, 0(x0)  — buf_b */
        0x83, 0x28, 0x00, 0x00,   /* lw a7, 0(x0)  — buf_c */
        0x63, 0x00, 0x00, 0x00,   /* vadd.vv v0, a1, a2 */
        0x23, 0x00, 0x08, 0x02,   /* sw v0, 0(a7) */
        0x67, 0x80, 0x00, 0x00,   /* ret */
    };
    
    conflux_kernel_t *kernel = conflux_kernel_create("vector_add", 42,
                                                    vec_add_instr,
                                                    sizeof(vec_add_instr));
    assert(kernel != NULL);
    
    /* 上传指令到设备 */
    uint64_t instr_addr;
    ret = conflux_cmd_builder_upload_kernel(builder, kernel, &instr_addr);
    assert(ret == CONFLUX_SUCCESS);
    CONFLUX_INFO("  Kernel 'vector_add' uploaded to 0x%lx (%zu bytes)",
                (unsigned long)instr_addr, sizeof(vec_add_instr));
    
    /* ============================================================
     * 阶段 7：设置内核参数（对应 clSetKernelArg）
     * ============================================================ */
    CONFLUX_INFO("--- Phase 7: Set Kernel Args ---");
    
    ret = conflux_kernel_set_arg(kernel, 0, sizeof(uint64_t), &buf_a, 0);
    assert(ret == CONFLUX_SUCCESS);
    
    ret = conflux_kernel_set_arg(kernel, 1, sizeof(uint64_t), &buf_b, 0);
    assert(ret == CONFLUX_SUCCESS);
    
    ret = conflux_kernel_set_arg(kernel, 2, sizeof(uint64_t), &buf_c, 0);
    assert(ret == CONFLUX_SUCCESS);
    
    conflux_kernel_dump(kernel);
    
    /* ============================================================
     * 阶段 8：执行内核（对应 clEnqueueNDRangeKernel）
     * ============================================================ */
    CONFLUX_INFO("--- Phase 8: Launch Kernel ---");
    
    conflux_launch_request_t launch_req = {
        .kernel      = kernel,
        .work_dim    = 1,
        .global_size = {1024, 1, 1},
        .local_size  = {64, 1, 1},
    };
    
    conflux_event_t *launch_ev = NULL;
    t0 = conflux_get_time_ns();
    ret = conflux_cmd_builder_launch(builder, &launch_req, &launch_ev);
    assert(ret == CONFLUX_SUCCESS);
    
    conflux_event_wait(launch_ev, 0);
    t1 = conflux_get_time_ns();
    conflux_perf_record_wait(t1 - t0);
    
    CONFLUX_INFO("  Kernel completed in %lu ns", (unsigned long)(t1 - t0));
    conflux_event_destroy(launch_ev);
    
    /* ============================================================
     * 阶段 9：读回结果（对应 clEnqueueReadBuffer）
     * ============================================================ */
    CONFLUX_INFO("--- Phase 9: Read Results ---");
    
    conflux_copy_request_t read_req = {
        .src_addr = buf_c,
        .dst_addr = 0,      /* 主机内存 */
        .size     = 4096,
    };
    
    conflux_event_t *read_ev = NULL;
    ret = conflux_cmd_builder_copy(builder, &read_req, &read_ev);
    assert(ret == CONFLUX_SUCCESS);
    conflux_event_wait(read_ev, 0);
    conflux_event_destroy(read_ev);
    
    CONFLUX_INFO("  Results read back");
    
    /* ============================================================
     * 阶段 10：批量操作（测试管道深度）
     * ============================================================ */
    CONFLUX_INFO("--- Phase 10: Batch Operations ---");
    
    #define BATCH_SIZE 10
    
    conflux_event_t *events[BATCH_SIZE];
    
    for (int i = 0; i < BATCH_SIZE; i++) {
        conflux_copy_request_t req = {
            .src_addr = buf_a,
            .dst_addr = buf_b,
            .size     = 256,
        };
        ret = conflux_cmd_builder_copy(builder, &req, &events[i]);
        assert(ret == CONFLUX_SUCCESS);
    }
    
    /* 等全部完成 */
    for (int i = 0; i < BATCH_SIZE; i++) {
        t0 = conflux_get_time_ns();
        conflux_event_wait(events[i], 0);
        t1 = conflux_get_time_ns();
        conflux_perf_record_wait(t1 - t0);
        conflux_event_destroy(events[i]);
    }
    
    CONFLUX_INFO("  %d copies completed", BATCH_SIZE);
    
    /* ============================================================
     * 阶段 11：释放资源
     * ============================================================ */
    CONFLUX_INFO("--- Phase 11: Cleanup ---");
    
    /* 释放缓冲区 */
    conflux_cmd_builder_free(builder, buf_a, 4096, NULL);
    conflux_cmd_builder_free(builder, buf_b, 4096, NULL);
    conflux_cmd_builder_free(builder, buf_c, 4096, NULL);
    
    /* 释放内核 */
    conflux_kernel_destroy(kernel);
    
    /* 释放构建器 */
    conflux_cmd_builder_dump(builder);
    conflux_cmd_builder_destroy(builder);
    
    /* 停止并释放队列 */
    conflux_queue_stop_consumer(queue);
    conflux_queue_dump(queue);
    conflux_queue_destroy(queue);
    
    /* 关闭设备 */
    conflux_platform_close_all();
    
    /* ============================================================
     * 阶段 12：性能统计
     * ============================================================ */
    CONFLUX_INFO("--- Phase 12: Performance Summary ---");
    conflux_perf_print();
    
    /* dump 日志环（崩溃诊断） */
    CONFLUX_INFO("--- Log Ring Dump ---");
    conflux_log_dump_ring(stdout);
    
    /* 清理平台 */
    conflux_platform_destroy();
    
    CONFLUX_INFO("=== All Phases PASSED ===");
    
    return 0;
}