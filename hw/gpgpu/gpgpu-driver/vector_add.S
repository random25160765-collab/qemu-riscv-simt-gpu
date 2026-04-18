/*
 * 向量加法 GPU Kernel (RISC-V 汇编)
 * 编译: riscv64-linux-gnu-gcc -c vector_add.S -o vector_add.o
 * 链接: riscv64-linux-gnu-ld -T kernel.ld vector_add.o -o vector_add.elf
 * 提取: riscv64-linux-gnu-objcopy -O binary vector_add.elf vector_add.bin
 *
 * GPU 核心内部 CTRL 地址空间 (0x80000000+):
 *   - 主机侧通过 MMIO (0x1000, 0x1010, 0x2000) 可读写
 *   - GPU 核心通过内部 CTRL 地址空间 (0x80000000+) 只读访问
 */

.section .text
.globl _start

_start:
    /*
     * 寄存器说明:
     * GPU 核心内部 CTRL 地址映射:
     * 0x80000000: THREAD_ID_X
     * 0x80000010: BLOCK_ID_X
     *
     * 数据在 VRAM 中的地址（与测试程序约定）:
     * A: 0x00100000
     * B: 0x00200000
     * C: 0x00300000
     *
     * 假设每个 Block 固定 256 线程
     */

    /* 读取 thread_id_x */
    li   t0, 0x80000000
    lw   t1, 0(t0)          # t1 = threadIdx.x

    /* 读取 block_id_x */
    li   t0, 0x80000010
    lw   t2, 0(t0)          # t2 = blockIdx.x

    /* 计算全局 ID = blockIdx.x * 256 + threadIdx.x */
    li   t3, 256
    mul  t2, t2, t3         # t2 = blockIdx.x * 256
    add  t1, t2, t1         # t1 = global_id

    /* 计算数组偏移: global_id * 4 (float 是 4 字节) */
    slli t1, t1, 2          # t1 = offset in bytes

    /* 加载 A[global_id] */
    li   t0, 0x00100000
    add  t0, t0, t1
    flw  f0, 0(t0)          # f0 = A[global_id]

    /* 加载 B[global_id] */
    li   t0, 0x00200000
    add  t0, t0, t1
    flw  f1, 0(t0)          # f1 = B[global_id]

    /* C = A + B */
    fadd.s f2, f0, f1

    /* 存储到 C[global_id] */
    li   t0, 0x00300000
    add  t0, t0, t1
    fsw  f2, 0(t0)          # C[global_id] = f2

    /* 同步 barrier */
    li   t0, 0x80002000
    li   t1, 1
    sw   t1, 0(t0)          # 写 barrier 寄存器

    /* 完成，无限循环等待复位 */
1:  j    1b

    /* 确保 4 字节对齐 */
    .align 2