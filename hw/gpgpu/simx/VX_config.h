// 从 Vortex hw/rtl/VX_config.vh 翻译而来的 C 版本
// 固定配置：XLEN=64, 4 warps, 4 threads, 1 core
// 架构参数（num_cores/num_warps/num_threads）由 Arch() 构造函数在运行时传入
#pragma once

#ifndef MIN
#define MIN(x, y)   (((x) < (y)) ? (x) : (y))
#endif
#ifndef MAX
#define MAX(x, y)   (((x) > (y)) ? (x) : (y))
#endif
#ifndef UP
#define UP(x)   (((x) != 0) ? (x) : 1)
#endif

// ISA 扩展
#define EXT_M_ENABLE
#define EXT_F_ENABLE
#define EXT_D_ENABLE
#define EXT_ZICOND_ENABLE

// XLEN
#define XLEN_64
#define XLEN 64

// 浮点宽度
#define FLEN_64
#define FLEN 64
#define FPU_RV64F

// 向量寄存器宽度
#define VLEN (4 * XLEN)

// 默认拓扑（运行时由 Arch() 覆盖）
#define NUM_CLUSTERS  1
#define NUM_CORES     1
#define NUM_WARPS     4
#define NUM_THREADS   4
#define NUM_BARRIERS  UP(NUM_WARPS / 2)
#define SOCKET_SIZE   MIN(4, NUM_CORES)

// 内存块参数
#define MEM_BLOCK_SIZE      64
#define MEM_ADDR_WIDTH      48
#define L1_LINE_SIZE        MEM_BLOCK_SIZE
#define L2_LINE_SIZE        MEM_BLOCK_SIZE
#define L3_LINE_SIZE        MEM_BLOCK_SIZE
#define MEM_PAGE_SIZE       4096
#define MEM_PAGE_LOG2_SIZE  12

// 平台内存
#define PLATFORM_MEMORY_NUM_BANKS   2
#define PLATFORM_MEMORY_ADDR_WIDTH  48
#define PLATFORM_MEMORY_DATA_SIZE   64
#define PLATFORM_MEMORY_INTERLEAVE  1

// 地址布局（XLEN=64）
#define STACK_BASE_ADDR     0x1FFFF0000ULL
#define STARTUP_ADDR        0x080000000ULL
#define USER_BASE_ADDR      0x000010000ULL
#define IO_BASE_ADDR        0x000000040ULL
#define PAGE_TABLE_BASE_ADDR 0x0F0000000ULL
#define IO_END_ADDR         USER_BASE_ADDR
#define IO_COUT_ADDR        IO_BASE_ADDR
#define IO_COUT_SIZE        64
#define IO_MPM_ADDR         (IO_COUT_ADDR + IO_COUT_SIZE)

// 局部内存
#define LMEM_LOG_SIZE   14
#define LMEM_BASE_ADDR  STACK_BASE_ADDR

// 栈
#define STACK_LOG2_SIZE 13

// 仿真参数
#define RESET_DELAY     8
#define STALL_TIMEOUT   100000

// 浮点单元（用 DSP 路径，无 DPI）
#define FPU_DSP

// 调试
#define DEBUG_LEVEL 3

// SIMD / Issue 宽度
#define SIMD_WIDTH      NUM_THREADS
#define ISSUE_WIDTH     UP(NUM_WARPS / 16)
#define NUM_OPCS        UP(NUM_WARPS / (4 * ISSUE_WIDTH))

// 寄存器 bank
#define NUM_GPR_BANKS   4
#define NUM_VGPR_BANKS  2

// ALU/FPU/LSU/SFU
#define NUM_ALU_LANES   SIMD_WIDTH
#define NUM_ALU_BLOCKS  ISSUE_WIDTH
#define NUM_FPU_LANES   SIMD_WIDTH
#define NUM_FPU_BLOCKS  ISSUE_WIDTH
#define NUM_LSU_LANES   SIMD_WIDTH
#define NUM_LSU_BLOCKS  1
#define NUM_SFU_LANES   SIMD_WIDTH
#define NUM_SFU_BLOCKS  1
#define NUM_VPU_LANES   SIMD_WIDTH
#define NUM_VPU_BLOCKS  ISSUE_WIDTH
#define NUM_TCU_LANES   NUM_THREADS
#define NUM_TCU_BLOCKS  ISSUE_WIDTH

// LSU 行宽
#define LSU_LINE_SIZE   MIN(NUM_LSU_LANES * (XLEN / 8), L1_LINE_SIZE)
#define LSUQ_IN_SIZE    (2 * (SIMD_WIDTH / NUM_LSU_LANES))
#define LSUQ_OUT_SIZE   (2 * (SIMD_WIDTH / NUM_LSU_LANES))

// L1 cache 禁用时的内存端口数
// DCACHE_WORD_SIZE = LSU_LINE_SIZE = MIN(4*8, 64) = 32
// DCACHE_CHANNELS  = UP((4*8)/32) = 1
// DCACHE_NUM_REQS  = NUM_LSU_BLOCKS * DCACHE_CHANNELS = 1
// L1_MEM_PORTS = MIN(DCACHE_NUM_REQS, PLATFORM_MEMORY_NUM_BANKS) = MIN(1,2) = 1
#define L1_DISABLE
#define L1_MEM_PORTS    1

// I-Cache（禁用）
#define ICACHE_DISABLE
#define NUM_ICACHES     0
#define ICACHE_ENABLED  0
#define ICACHE_SIZE     8192
#define ICACHE_NUM_WAYS 2
#define ICACHE_CRSQ_SIZE 2
#define ICACHE_MREQ_SIZE 4
#define ICACHE_MRSQ_SIZE 0
#define ICACHE_MSHR_SIZE 8
#define ICACHE_MEM_PORTS 1
#define ICACHE_REPL_POLICY 0

// D-Cache（禁用）
#define DCACHE_DISABLE
#define NUM_DCACHES     0
#define DCACHE_ENABLED  0
#define DCACHE_SIZE     8192
#define DCACHE_NUM_WAYS 2
#define DCACHE_CRSQ_SIZE 2
#define DCACHE_MREQ_SIZE 4
#define DCACHE_MRSQ_SIZE 0
#define DCACHE_MSHR_SIZE 8
#define DCACHE_NUM_BANKS 1
#define DCACHE_WRITEBACK 0
#define DCACHE_DIRTYBYTES DCACHE_WRITEBACK
#define DCACHE_REPL_POLICY 0

// 局部内存（启用）
#define LMEM_ENABLE
#define LMEM_ENABLED    1
#define LMEM_NUM_BANKS  2

// GBAR（启用）
#define GBAR_ENABLED    1

// L2（禁用）
#define L2_ENABLED      0
#define L2_CACHE_SIZE   131072
#define L2_NUM_BANKS    2
#define L2_NUM_WAYS     4
#define L2_CRSQ_SIZE    2
#define L2_MSHR_SIZE    16
#define L2_MREQ_SIZE    4
#define L2_MRSQ_SIZE    4
#define L2_WRITEBACK    0
#define L2_DIRTYBYTES   L2_WRITEBACK
#define L2_REPL_POLICY  1
#define L2_MEM_PORTS    MIN(L2_NUM_BANKS, PLATFORM_MEMORY_NUM_BANKS)

// L3（禁用）
#define L3_ENABLED      0
#define L3_CACHE_SIZE   2097152
#define L3_NUM_BANKS    2
#define L3_NUM_WAYS     8
#define L3_CRSQ_SIZE    2
#define L3_MSHR_SIZE    16
#define L3_MREQ_SIZE    4
#define L3_MRSQ_SIZE    4
#define L3_WRITEBACK    0
#define L3_DIRTYBYTES   L3_WRITEBACK
#define L3_REPL_POLICY  1
#define L3_MEM_PORTS    MIN(L3_NUM_BANKS, PLATFORM_MEMORY_NUM_BANKS)

// ISA 扩展标志（运行时用）
#define EXT_M_ENABLED   1
#define EXT_F_ENABLED   1
#define EXT_D_ENABLED   1
#define EXT_A_ENABLED   0
#define EXT_C_ENABLED   0
#define EXT_V_ENABLED   0
#define EXT_ZICOND_ENABLED 1
#define EXT_TCU_ENABLED 0

// ISA 编码常量
#define ISA_STD_A   0
#define ISA_STD_C   2
#define ISA_STD_D   3
#define ISA_STD_E   4
#define ISA_STD_F   5
#define ISA_STD_H   7
#define ISA_STD_I   8
#define ISA_STD_N   13
#define ISA_STD_Q   16
#define ISA_STD_S   18
#define ISA_STD_V   21
#define ISA_EXT_ICACHE  0
#define ISA_EXT_DCACHE  1
#define ISA_EXT_L2CACHE 2
#define ISA_EXT_L3CACHE 3
#define ISA_EXT_LMEM    4
#define ISA_EXT_ZICOND  5
#define ISA_EXT_TCU     6

#define MISA_EXT  ((ICACHE_ENABLED  << ISA_EXT_ICACHE)  \
                 | (DCACHE_ENABLED  << ISA_EXT_DCACHE)  \
                 | (L2_ENABLED      << ISA_EXT_L2CACHE) \
                 | (L3_ENABLED      << ISA_EXT_L3CACHE) \
                 | (LMEM_ENABLED    << ISA_EXT_LMEM)    \
                 | (EXT_ZICOND_ENABLED << ISA_EXT_ZICOND) \
                 | (EXT_TCU_ENABLED << ISA_EXT_TCU))

#define MISA_STD  ((EXT_A_ENABLED << 0)  \
                 | (EXT_C_ENABLED << 2)  \
                 | (EXT_D_ENABLED << 3)  \
                 | (EXT_F_ENABLED << 5)  \
                 | (1             << 8)  \
                 | (EXT_M_ENABLED << 12) \
                 | (1             << 20) \
                 | (EXT_V_ENABLED << 21) \
                 | (1             << 23))

#define VENDOR_ID           0
#define ARCHITECTURE_ID     0
#define IMPLEMENTATION_ID   0

// FPU 延迟
#define LATENCY_FMA     4
#define LATENCY_FDIV    15
#define LATENCY_FSQRT   15
#define LATENCY_FCVT    4
#define LATENCY_FNCP    2

// FPU 比例
#define FMA_PE_RATIO    2
#define FDIV_PE_RATIO   2
#define FSQRT_PE_RATIO  2
#define FCVT_PE_RATIO   2
#define FNCP_PE_RATIO   2
#define FPUQ_SIZE       8

// IBUF
#define IBUF_SIZE       2
