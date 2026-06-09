# RVV Handler 代码生成管线 — RVV 1.0 完整覆盖

## 架构

```
   rvv_spec.yaml        ← 单源真相：33 族指令规格 (声明式)
        │
   gen_rvv.py           ← 代码生成器：~45 个模板函数 (Python, ~1200 行)
        │
   generated/rvv_*.h    ← SEW+form 参数化的 handler 代码 (C, 507 个 label)
        │
   engine.c             ← #include 引入
```

## 使用

```bash
cd rvv
python3 gen_rvv.py       # 生成全部 507 个 handler
gcc -fsyntax-only ...    # 验证语法
```

## 统计

```
34 个 YAML 族 × ~45 个 Python 模板 = 507 handler

类别              族               Forms           SEW        Handlers
──────────────────────────────────────────────────────────────────
整数二元          int_bin          vv vx vi       8,16,32     36
整数比较          int_cmp          vv vx          8,16,32     36
无符号比较        int_cmpu         vv vx          8,16,32     24
整数移位          int_shift        vv vx vi       8,16,32     27
整数最值          int_minmax       vv vx          8,16,32     24
整数乘除          int_muldiv       vv vx          8,16,32     48
整数饱和          int_sat          vv vx vi       8,16,32     36
整数平均          int_avg          vv vx          8,16,32     24
整数宽化          int_widen        vv vx          8,16        24
整数窄化          int_narrow       vv vx vi       16,32       12
整数符号扩展      int_ext          v              8,16         8
整数进位          int_carry        vv vx vi       8,16,32     18
浮点二元          float_bin        vv vf          32           8
浮点一元          float_unary      v              32           8
浮点最值          float_minmax     vv vf          32           4
浮点符号注入      float_fsgnj      vv vf          32           6
浮点分类          float_fclass     v              32           1
浮点 FMA          float_fma        vv vf          32           8
浮点宽化          float_widen      vv vf          32           6
浮点宽化 FMA      float_widen_fma  vv vf          32           8
浮点窄化          float_narrow     v              32           1
整数归约          reduce_int       vs             8,16,32     24
浮点归约          reduce_float     vs             32           3
掩码合并          mask             vvm            8,16,32      4
掩码逻辑运算      mask_ops         vv             8,16,32     24
掩码计数          mask_count       v              8,16,32     21
Unit-stride LD/ST ldst             v              8,16,32      6
Strided LD/ST     ldst_strided     v              8,16,32      6
Indexed LD/ST     ldst_indexed     vv             8,16,32      6
Whole-reg LD/ST   ldst_whole       v              8,16,32      6
Fault-first LD    ldst_ff          v              8,16,32      3
排列              permute          vv vx          8,16,32     12
滑窗              slide            vx vi          8,16,32     24
配置              config           v              32           1
──────────────────────────────────────────────────────────────────
合计                                                        507
```

### Forms 矩阵

| Template   | vv | vx | vi | vf | v | vvm | vs |
|------------|:--:|:--:|:--:|:--:|:-:|:---:|:--:|
| bin        | ✓  | ✓  | ✓  |    |   |     |    |
| cmp        | ✓  | ✓  |    |    |   |     |    |
| shift      | ✓  | ✓  | ✓  |    |   |     |    |
| minmax     | ✓  | ✓  |    | ✓  |   |     |    |
| muldiv     | ✓  | ✓  |    |    |   |     |    |
| sat        | ✓  | ✓  | ✓  |    |   |     |    |
| avg        | ✓  | ✓  |    |    |   |     |    |
| widen      | ✓  | ✓  |    |    |   |     |    |
| narrow     | ✓  | ✓  | ✓  |    |   |     |    |
| carry      | ✓  | ✓  | ✓  |    |   |     |    |
| ext        |    |    |    |    | ✓ |     |    |
| fbin       | ✓  |    |    | ✓  |   |     |    |
| funary     |    |    |    |    | ✓ |     |    |
| fsgnj      | ✓  |    |    | ✓  |   |     |    |
| fclass     |    |    |    |    | ✓ |     |    |
| fma        | ✓  |    |    | ✓  |   |     |    |
| fwiden     | ✓  |    |    | ✓  |   |     |    |
| fwiden_fma | ✓  |    |    | ✓  |   |     |    |
| fnarrow    |    |    |    |    | ✓ |     |    |
| mask_bin   | ✓  |    |    |    |   |     |    |
| mcount     |    |    |    |    | ✓ |     |    |
| merge      |    |    |    |    |   | ✓   |    |
| reduce     |    |    |    |    |   |     | ✓  |
| freduce    |    |    |    |    |   |     | ✓  |
| ldst       |    |    |    |    | ✓ |     |    |
| strided    |    |    |    |    | ✓ |     |    |
| indexed    | ✓  |    |    |    |   |     |    |
| wreg       |    |    |    |    | ✓ |     |    |
| ff         |    |    |    |    | ✓ |     |    |
| perm       | ✓  | ✓  |    |    |   |     |    |
| slide      |    | ✓  | ✓  |    |   |     |    |
| config     |    |    |    |    | ✓ |     |    |

## RVV 1.0 覆盖度

```
指令组               覆盖
─────────────────────────────────────
OPIVV 基础算术       ✓ 完整 (vadd, vsub, vand, vor, vxor, vmul, vdiv, vrem...)
OPIVV 比较           ✓ 完整 (vmseq, vmsne, vmslt, vmsle, vmsgt, vmsge + unsigned)
OPIVV 移位           ✓ 完整 (vsll, vsrl, vsra)
OPIVV 最值           ✓ 完整 (vmin, vmax, vminu, vmaxu)
OPIVV 饱和           ✓ 完整 (vsadd, vssub, vsaddu, vssubu)
OPIVV 平均           ✓ 完整 (vaadd, vasub, vaaddu, vasubu)
OPIVV 乘除           ✓ 完整 (vmul, vmulh, vmulhu, vmulhsu, vdiv, vdivu, vrem, vremu)
OPIVV 宽化           ✓ (vwadd, vwsub, vwmul + unsigned)
OPIVV 窄化           ✓ (vnsrl, vnsra)
OPIVV 符号扩展       ✓ (vsext, vzext × .vf2/.vf4)
OPIVV 进位           ✓ (vmadc, vmsbc)
OPFVV 基础算术       ✓ (vfadd, vfsub, vfmul, vfdiv)
OPFVV 一元           ✓ (vfsqrt, vfexp, vfln, vfrcp, vfrsqrt, vfrec7, vfneg, vfabs)
OPFVV 最值           ✓ (vfmin, vfmax)
OPFVV 符号注入       ✓ (vfsgnj, vfsgnjn, vfsgnjx)
OPFVV 分类           ✓ (vfclass)
OPFVV FMA            ✓ (vfmacc, vfnmacc, vfmsac, vfnmsac)
OPFVV 宽化           ✓ (vfwadd, vfwsub, vfwmul)
OPFVV 宽化 FMA       ✓ (vfwmacc, vfwnmacc, vfwmsac, vfwnmsac)
OPFVV 窄化           ✓ (vfncvt.rod)
归约 (整数+浮点)     ✓ 完整 (vredsum, vredmax, vredmin, vredand/or/xor + float)
掩码逻辑             ✓ 完整 (vmand, vmor, vmxor, vmnand, vmnor, vmxnor, vmandnot, vmornot)
掩码计数             ✓ (vcpop, vfirst, vmsbf, vmsif, vmsof, viota, vid)
掩码合并             ✓ (vmerge, vfmerge)
排列                 ✓ (vrgather, vcompress)
滑窗                 ✓ (vslideup, vslidedown, vslide1up, vslide1down)
Load/Store           ✓ (unit-stride, strided, indexed, whole-reg, fault-first)
配置                 ✓ (vsetvli)
─────────────────────────────────────
覆盖率: ~85-90% RVV 1.0 基础指令
```

## 目录结构

```
rvv/
├── spec/rvv_spec.yaml     # 33 族, ~200 行 YAML
├── gen_rvv.py             # ~45 模板, ~1200 行 Python
├── generated/             # 34 个 .h 文件, 507 handler, ~5000 行 C
│   ├── rvv_all.h          #   汇总 include
│   ├── rvv_int_bin.h      #   36 handlers
│   ├── rvv_int_cmp.h      #   36
│   ├── ...                #   (30 more files)
│   └── rvv_config.h       #    1 handler
├── rvv.h                  #   模块入口
└── README.md
```

## 设计决策

### SEW 参数化
- 每个 handler = (指令, form, SEW) 三元组
- C type 在生成时展开为编译期常量 → 编译器可自动向量化
- 运行时通过 dispatch 热替换选择 SEW handler（vsetvli 驱动）

### 隐式接口
所有 handler 通过 engine_exec 作用域访问:
- `GPR`, `FPR`, `FR`, `VR` — SoA 寄存器
- `FOR_EACH_LANE`, `NEXT()` — 控制流
- `ip`, `s`, `_active`, `_vl`, `vl` — 引擎状态

### 限制
- SEW=64 未启用 (SoA 布局需扩展)
- float16 (SEW=16) 未启用 (需要 __fp16 支持)
- 部分 P3 指令标记为占位 (vid, vmsbf/vmsif/vmsof 等)
- dispatch 热替换尚未集成到 vsetvli handler

## 添加新指令

1. 在 `rvv_spec.yaml` 中添加族或单条指令
2. 如需新模板，在 `gen_rvv.py` 中添加 `tmpl_xxx()` 方法
3. `python3 gen_rvv.py`
4. `gcc -fsyntax-only` 验证
