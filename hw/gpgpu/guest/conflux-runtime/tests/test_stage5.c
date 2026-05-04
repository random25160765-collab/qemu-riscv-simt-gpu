#include "conflux_kernel.h"
#include <stdio.h>
#include <assert.h>

/* 模拟一段 GPU 指令 */
uint8_t dummy_instr[] = {0x00, 0x01, 0x02, 0x03, 0xDE, 0xAD, 0xBE, 0xEF};

int main(void) 
{
    printf("=== Stage 5: Kernel Object ===\n\n");
    
    /* 测试 1：创建/销毁 */
    printf("--- Test 1: Create and destroy ---\n");
    conflux_kernel_t *k = conflux_kernel_create("vector_add", 42,
                                               dummy_instr, sizeof(dummy_instr));
    assert(k != NULL);
    conflux_kernel_dump(k);
    conflux_kernel_destroy(k);
    printf("  OK\n\n");
    
    /* 测试 2：设置参数 */
    printf("--- Test 2: Set arguments ---\n");
    k = conflux_kernel_create("matmul", 7, dummy_instr, sizeof(dummy_instr));
    
    int buf_a = 0x1000;
    int buf_b = 0x2000;
    int buf_c = 0x3000;
    float alpha = 2.5f;
    
    /* 设置 4 个参数 */
    conflux_kernel_set_arg(k, 0, sizeof(int), &buf_a, 0);
    conflux_kernel_set_arg(k, 1, sizeof(int), &buf_b, 0);
    conflux_kernel_set_arg(k, 2, sizeof(int), &buf_c, 0);
    conflux_kernel_set_arg(k, 3, sizeof(float), &alpha, 0);
    /* 第 5 个参数：local memory */
    conflux_kernel_set_arg(k, 4, 256, NULL, 1);  /* local mem 不需要 host 指针 */
    
    assert(k->num_args == 5);
    conflux_kernel_dump(k);
    printf("  OK\n\n");
    
    /* 测试 3：设置工作维度 */
    printf("--- Test 3: Set NDRange ---\n");
    conflux_kernel_set_work_dim(k, 2);
    conflux_kernel_set_global_size(k, 0, 1024);
    conflux_kernel_set_global_size(k, 1, 768);
    conflux_kernel_set_local_size(k, 0, 16);
    conflux_kernel_set_local_size(k, 1, 16);
    
    assert(k->work_dim == 2);
    assert(k->global_size[0] == 1024);
    assert(k->global_size[1] == 768);
    assert(k->local_size[0] == 16);
    
    conflux_kernel_dump(k);
    printf("  OK\n\n");
    
    /* 测试 4：上传指令并打包成命令 */
    printf("--- Test 4: Upload binary and pack command ---\n");
    int ret = conflux_kernel_upload(k, 0xDEAD0000);
    assert(ret == CONFLUX_SUCCESS);
    
    conflux_cmd_t cmd;
    ret = conflux_kernel_pack_cmd(k, &cmd);
    assert(ret == CONFLUX_SUCCESS);
    assert(cmd.type == CONFLUX_CMD_KERNEL);
    assert(cmd.kernel_id == 7);
    assert(cmd.src_addr == 0xDEAD0000);
    assert((cmd.flags & 0x3) == 2);  /* work_dim == 2 */
    
    printf("  Cmd packed: type=%d, kernel_id=%u, src=0x%lx, size=%u\n",
           cmd.type, cmd.kernel_id, (unsigned long)cmd.src_addr, cmd.size);
    printf("  OK\n\n");
    
    /* 测试 5：参数超限 */
    printf("--- Test 5: Argument index exceeds max ---\n");
    ret = conflux_kernel_set_arg(k, 32, 4, NULL, 0);  /* 32 >= MAX(32) */
    assert(ret == CONFLUX_ERR_INVALID);
    printf("  Correctly rejected (ret=%d)\n", ret);
    printf("  OK\n\n");
    
    /* 测试 6：未上传就打包应失败 */
    printf("--- Test 6: Pack without upload should fail ---\n");
    conflux_kernel_t *k2 = conflux_kernel_create("no_upload", 1, dummy_instr, 8);
    ret = conflux_kernel_pack_cmd(k2, &cmd);
    assert(ret == CONFLUX_ERR_INVALID);
    printf("  Correctly rejected (ret=%d)\n", ret);
    conflux_kernel_destroy(k2);
    printf("  OK\n\n");
    
    /* 清理 */
    conflux_kernel_destroy(k);
    
    printf("=== Stage 5: ALL PASSED ===\n");
    return 0;
}