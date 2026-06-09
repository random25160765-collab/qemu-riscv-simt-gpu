#!/usr/bin/env python3
"""
gen_rvv.py — RVV 1.0 handler 代码生成器

从 rvv_spec.yaml 生成 SEW+form 参数化的 C handler 头文件。
每个 handler 是 engine_exec() 内的 computed-goto label。

模板命名: tmpl_{family.template}_{form}()
"""

import yaml, os, sys
from textwrap import dedent

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(SCRIPT_DIR)                     # inst/rvv/
SPEC_DIR   = os.path.join(SCRIPT_DIR)                        # spec file next to script
OUT_DIR    = os.path.join(BASE_DIR, "handles")               # inst/rvv/handles/

SUPPORTED_SEW = {8, 16, 32}

# ============================================================================
# 辅助
# ============================================================================

def vr_cast(ctype: str) -> str:
    return f"*({ctype} *)&VR"

class RVVGen:
    def __init__(self, spec_path: str):
        with open(spec_path) as f:
            self.spec = yaml.safe_load(f)
        self.types = self.spec["types"]

    def ctype(self, sew: int, itype: str) -> str:
        return self.types[f"sew{sew}"][itype]

    # ========================================================================
    # §1-4 整数二元/比较/移位/最值 (已有, 保持不变)
    # ========================================================================

    def tmpl_bin_vv(self, inst, sew):
        T = self.ctype(sew, inst["itype"])
        return dedent(f"""\
        rvv_{inst['name']}_vv_e{sew}:
        {{
            int vd = ip[-1].rd, vs1 = ip[-1].rs1, vs2 = ip[-1].rs2;
            uint32_t _vm = (ip[-1].inst >> 25) & 1;
            FOR_EACH_LANE
            {{
                if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1)))
                    {vr_cast(T)}(vd, _li) = {vr_cast(T)}(vs1, _li) {inst['op']} {vr_cast(T)}(vs2, _li);
                PC(_li) += 4;
            }}
            NEXT();
        }}
        """)

    def tmpl_bin_vx(self, inst, sew):
        T = self.ctype(sew, inst["itype"])
        return dedent(f"""\
        rvv_{inst['name']}_vx_e{sew}:
        {{
            int vd = ip[-1].rd, vs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
            {T} sc = *({T} *)&GPR(rs2, 0);
            uint32_t _vm = (ip[-1].inst >> 25) & 1;
            FOR_EACH_LANE
            {{
                if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1)))
                    {vr_cast(T)}(vd, _li) = {vr_cast(T)}(vs1, _li) {inst['op']} sc;
                PC(_li) += 4;
            }}
            NEXT();
        }}
        """)

    def tmpl_bin_vi(self, inst, sew):
        T = self.ctype(sew, inst["itype"])
        return dedent(f"""\
        rvv_{inst['name']}_vi_e{sew}:
        {{
            int vd = ip[-1].rd, vs1 = ip[-1].rs1;
            {T} imm = ({T})(int32_t)ip[-1].imm;
            uint32_t _vm = (ip[-1].inst >> 25) & 1;
            FOR_EACH_LANE
            {{
                if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1)))
                    {vr_cast(T)}(vd, _li) = {vr_cast(T)}(vs1, _li) {inst['op']} imm;
                PC(_li) += 4;
            }}
            NEXT();
        }}
        """)

    def tmpl_cmp_vv(self, inst, sew):
        T = self.ctype(sew, inst["itype"])
        return dedent(f"""\
        rvv_{inst['name']}_vv_e{sew}:
        {{
            int vd = ip[-1].rd, vs1 = ip[-1].rs1, vs2 = ip[-1].rs2;
            uint32_t _vm = (ip[-1].inst >> 25) & 1;
            FOR_EACH_LANE
            {{
                if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1)))
                    {vr_cast(T)}(vd, _li) = ({vr_cast(T)}(vs1, _li) {inst['op']} {vr_cast(T)}(vs2, _li)) ? -1 : 0;
                PC(_li) += 4;
            }}
            NEXT();
        }}
        """)

    def tmpl_cmp_vx(self, inst, sew):
        T = self.ctype(sew, inst["itype"])
        return dedent(f"""\
        rvv_{inst['name']}_vx_e{sew}:
        {{
            int vd = ip[-1].rd, vs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
            {T} sc = *({T} *)&GPR(rs2, 0);
            uint32_t _vm = (ip[-1].inst >> 25) & 1;
            FOR_EACH_LANE
            {{
                if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1)))
                    {vr_cast(T)}(vd, _li) = ({vr_cast(T)}(vs1, _li) {inst['op']} sc) ? -1 : 0;
                PC(_li) += 4;
            }}
            NEXT();
        }}
        """)

    def tmpl_shift_vv(self, inst, sew):
        T = self.ctype(sew, inst["itype"])
        UT = self.ctype(sew, "uint")
        mask = sew - 1
        return dedent(f"""\
        rvv_{inst['name']}_vv_e{sew}:
        {{
            int vd = ip[-1].rd, vs1 = ip[-1].rs1, vs2 = ip[-1].rs2;
            uint32_t _vm = (ip[-1].inst >> 25) & 1;
            FOR_EACH_LANE
            {{
                if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1))) {{
                    {UT} _shamt = {vr_cast(UT)}(vs2, _li) & {mask}U;
                    {vr_cast(T)}(vd, _li) = {vr_cast(T)}(vs1, _li) {inst['op']} _shamt;
                }}
                PC(_li) += 4;
            }}
            NEXT();
        }}
        """)

    def tmpl_shift_vx(self, inst, sew):
        T = self.ctype(sew, inst["itype"])
        UT = self.ctype(sew, "uint")
        mask = sew - 1
        return dedent(f"""\
        rvv_{inst['name']}_vx_e{sew}:
        {{
            int vd = ip[-1].rd, vs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
            {UT} sc = *({UT} *)&GPR(rs2, 0) & {mask}U;
            uint32_t _vm = (ip[-1].inst >> 25) & 1;
            FOR_EACH_LANE
            {{
                if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1)))
                    {vr_cast(T)}(vd, _li) = {vr_cast(T)}(vs1, _li) {inst['op']} sc;
                PC(_li) += 4;
            }}
            NEXT();
        }}
        """)

    def tmpl_shift_vi(self, inst, sew):
        T = self.ctype(sew, inst["itype"])
        UT = self.ctype(sew, "uint")
        mask = sew - 1
        return dedent(f"""\
        rvv_{inst['name']}_vi_e{sew}:
        {{
            int vd = ip[-1].rd, vs1 = ip[-1].rs1;
            {UT} imm = ({UT})((int32_t)ip[-1].imm) & {mask}U;
            uint32_t _vm = (ip[-1].inst >> 25) & 1;
            FOR_EACH_LANE
            {{
                if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1)))
                    {vr_cast(T)}(vd, _li) = {vr_cast(T)}(vs1, _li) {inst['op']} imm;
                PC(_li) += 4;
            }}
            NEXT();
        }}
        """)

    def tmpl_minmax_vv(self, inst, sew):
        T = self.ctype(sew, inst["itype"])
        return dedent(f"""\
        rvv_{inst['name']}_vv_e{sew}:
        {{
            int vd = ip[-1].rd, vs1 = ip[-1].rs1, vs2 = ip[-1].rs2;
            uint32_t _vm = (ip[-1].inst >> 25) & 1;
            FOR_EACH_LANE
            {{
                if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1))) {{
                    {T} a = {vr_cast(T)}(vs1, _li), b = {vr_cast(T)}(vs2, _li);
                    {vr_cast(T)}(vd, _li) = a {inst['cmp']} b ? a : b;
                }}
                PC(_li) += 4;
            }}
            NEXT();
        }}
        """)

    def tmpl_minmax_vx(self, inst, sew):
        T = self.ctype(sew, inst["itype"])
        return dedent(f"""\
        rvv_{inst['name']}_vx_e{sew}:
        {{
            int vd = ip[-1].rd, vs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
            {T} sc = *({T} *)&GPR(rs2, 0);
            uint32_t _vm = (ip[-1].inst >> 25) & 1;
            FOR_EACH_LANE
            {{
                if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1))) {{
                    {T} a = {vr_cast(T)}(vs1, _li);
                    {vr_cast(T)}(vd, _li) = a {inst['cmp']} sc ? a : sc;
                }}
                PC(_li) += 4;
            }}
            NEXT();
        }}
        """)

    def tmpl_minmax_vf(self, inst, sew):
        """浮点最值 with FPR scalar"""
        T = self.ctype(sew, inst["itype"])
        return dedent(f"""\
        rvv_{inst['name']}_vf_e{sew}:
        {{
            int vd = ip[-1].rd, vs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
            {T} sc = FR(rs2, 0);
            uint32_t _vm = (ip[-1].inst >> 25) & 1;
            FOR_EACH_LANE
            {{
                if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1))) {{
                    {T} a = {vr_cast(T)}(vs1, _li);
                    {vr_cast(T)}(vd, _li) = a {inst['cmp']} sc ? a : sc;
                }}
                PC(_li) += 4;
            }}
            NEXT();
        }}
        """)

    def tmpl_merge_vvm(self, inst, sew):
        T = self.ctype(sew, inst["itype"])
        return dedent(f"""\
        rvv_{inst['name']}_vvm_e{sew}:
        {{
            int vd = ip[-1].rd, vs1 = ip[-1].rs1, vs2 = ip[-1].rs2;
            FOR_EACH_LANE
            {{
                if ((uint32_t)_li < _vl)
                    {vr_cast(T)}(vd, _li) = ({vr_cast("uint32_t")}(0, _li) & 1)
                        ? {vr_cast(T)}(vs2, _li)
                        : {vr_cast(T)}(vs1, _li);
                PC(_li) += 4;
            }}
            NEXT();
        }}
        """)

    # ========================================================================
    # §5 整数乘除
    # ========================================================================

    def tmpl_muldiv_vv(self, inst, sew):
        T = self.ctype(sew, inst["itype"])
        op = inst["op"]
        n = inst["name"]
        if op in ("mh", "mhsu"):
            W = self.ctype(sew * 2, inst["itype"]) if op == "mh" else self.ctype(sew * 2, "int")
            return dedent(f"""\
            rvv_{n}_vv_e{sew}:
            {{
                int vd = ip[-1].rd, vs1 = ip[-1].rs1, vs2 = ip[-1].rs2;
                uint32_t _vm = (ip[-1].inst >> 25) & 1;
                FOR_EACH_LANE
                {{
                    if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1)))
                        {vr_cast(T)}(vd, _li) = ({T})(({W})({T}){vr_cast(T)}(vs1, _li) * ({W}){vr_cast(T)}(vs2, _li) >> {sew});
                    PC(_li) += 4;
                }}
                NEXT();
            }}
            """)
        elif op in ("/", "%"):
            return dedent(f"""\
            rvv_{n}_vv_e{sew}:
            {{
                int vd = ip[-1].rd, vs1 = ip[-1].rs1, vs2 = ip[-1].rs2;
                uint32_t _vm = (ip[-1].inst >> 25) & 1;
                FOR_EACH_LANE
                {{
                    if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1))) {{
                        {T} b = {vr_cast(T)}(vs2, _li);
                        {vr_cast(T)}(vd, _li) = b ? ({T})({vr_cast(T)}(vs1, _li) {op} b) : ({T})-1;
                    }}
                    PC(_li) += 4;
                }}
                NEXT();
            }}
            """)
        else:
            return dedent(f"""\
            rvv_{n}_vv_e{sew}:
            {{
                int vd = ip[-1].rd, vs1 = ip[-1].rs1, vs2 = ip[-1].rs2;
                uint32_t _vm = (ip[-1].inst >> 25) & 1;
                FOR_EACH_LANE
                {{
                    if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1)))
                        {vr_cast(T)}(vd, _li) = {vr_cast(T)}(vs1, _li) {op} {vr_cast(T)}(vs2, _li);
                    PC(_li) += 4;
                }}
                NEXT();
            }}
            """)

    def tmpl_muldiv_vx(self, inst, sew):
        T = self.ctype(sew, inst["itype"])
        op = inst["op"]
        n = inst["name"]
        if op in ("mh", "mhsu"):
            W = self.ctype(sew * 2, inst["itype"]) if op == "mh" else self.ctype(sew * 2, "int")
            return dedent(f"""\
            rvv_{n}_vx_e{sew}:
            {{
                int vd = ip[-1].rd, vs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
                {T} sc = *({T} *)&GPR(rs2, 0);
                uint32_t _vm = (ip[-1].inst >> 25) & 1;
                FOR_EACH_LANE
                {{
                    if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1)))
                        {vr_cast(T)}(vd, _li) = ({T})(({W})({T}){vr_cast(T)}(vs1, _li) * ({W})sc >> {sew});
                    PC(_li) += 4;
                }}
                NEXT();
            }}
            """)
        elif op in ("/", "%"):
            return dedent(f"""\
            rvv_{n}_vx_e{sew}:
            {{
                int vd = ip[-1].rd, vs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
                {T} sc = *({T} *)&GPR(rs2, 0);
                uint32_t _vm = (ip[-1].inst >> 25) & 1;
                FOR_EACH_LANE
                {{
                    if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1)))
                        {vr_cast(T)}(vd, _li) = sc ? ({T})({vr_cast(T)}(vs1, _li) {op} sc) : ({T})-1;
                    PC(_li) += 4;
                }}
                NEXT();
            }}
            """)
        else:
            return dedent(f"""\
            rvv_{n}_vx_e{sew}:
            {{
                int vd = ip[-1].rd, vs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
                {T} sc = *({T} *)&GPR(rs2, 0);
                uint32_t _vm = (ip[-1].inst >> 25) & 1;
                FOR_EACH_LANE
                {{
                    if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1)))
                        {vr_cast(T)}(vd, _li) = {vr_cast(T)}(vs1, _li) {op} sc;
                    PC(_li) += 4;
                }}
                NEXT();
            }}
            """)

    # ========================================================================
    # §6 整数饱和
    # ========================================================================

    def _sat_limits(self, sew, itype):
        if itype == "uint":
            return f"0", f"(uint{sew}_t)(-1)"
        else:
            return f"INT{sew}_MIN", f"INT{sew}_MAX"

    def tmpl_sat_vv(self, inst, sew):
        T = self.ctype(sew, inst["itype"])
        lo, hi = self._sat_limits(sew, inst["itype"])
        W = self.ctype(sew * 2, inst["itype"])
        return dedent(f"""\
        rvv_{inst['name']}_vv_e{sew}:
        {{
            int vd = ip[-1].rd, vs1 = ip[-1].rs1, vs2 = ip[-1].rs2;
            uint32_t _vm = (ip[-1].inst >> 25) & 1;
            FOR_EACH_LANE
            {{
                if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1))) {{
                    {W} r = ({W}){vr_cast(T)}(vs1, _li) {inst['op']} ({W}){vr_cast(T)}(vs2, _li);
                    {vr_cast(T)}(vd, _li) = r > ({W})({hi}) ? ({T})({hi}) : r < ({W})({lo}) ? ({T})({lo}) : ({T})r;
                }}
                PC(_li) += 4;
            }}
            NEXT();
        }}
        """)

    def tmpl_sat_vx(self, inst, sew):
        T = self.ctype(sew, inst["itype"])
        lo, hi = self._sat_limits(sew, inst["itype"])
        W = self.ctype(sew * 2, inst["itype"])
        return dedent(f"""\
        rvv_{inst['name']}_vx_e{sew}:
        {{
            int vd = ip[-1].rd, vs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
            {T} sc = *({T} *)&GPR(rs2, 0);
            uint32_t _vm = (ip[-1].inst >> 25) & 1;
            FOR_EACH_LANE
            {{
                if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1))) {{
                    {W} r = ({W}){vr_cast(T)}(vs1, _li) {inst['op']} ({W})sc;
                    {vr_cast(T)}(vd, _li) = r > ({W})({hi}) ? ({T})({hi}) : r < ({W})({lo}) ? ({T})({lo}) : ({T})r;
                }}
                PC(_li) += 4;
            }}
            NEXT();
        }}
        """)

    def tmpl_sat_vi(self, inst, sew):
        T = self.ctype(sew, inst["itype"])
        lo, hi = self._sat_limits(sew, inst["itype"])
        W = self.ctype(sew * 2, inst["itype"])
        return dedent(f"""\
        rvv_{inst['name']}_vi_e{sew}:
        {{
            int vd = ip[-1].rd, vs1 = ip[-1].rs1;
            {T} imm = ({T})(int32_t)ip[-1].imm;
            uint32_t _vm = (ip[-1].inst >> 25) & 1;
            FOR_EACH_LANE
            {{
                if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1))) {{
                    {W} r = ({W}){vr_cast(T)}(vs1, _li) {inst['op']} ({W})imm;
                    {vr_cast(T)}(vd, _li) = r > ({W})({hi}) ? ({T})({hi}) : r < ({W})({lo}) ? ({T})({lo}) : ({T})r;
                }}
                PC(_li) += 4;
            }}
            NEXT();
        }}
        """)

    # ========================================================================
    # §7 整数平均
    # ========================================================================

    def tmpl_avg_vv(self, inst, sew):
        T = self.ctype(sew, inst["itype"])
        W = self.ctype(sew * 2, inst["itype"])
        return dedent(f"""\
        rvv_{inst['name']}_vv_e{sew}:
        {{
            int vd = ip[-1].rd, vs1 = ip[-1].rs1, vs2 = ip[-1].rs2;
            uint32_t _vm = (ip[-1].inst >> 25) & 1;
            FOR_EACH_LANE
            {{
                if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1)))
                    {vr_cast(T)}(vd, _li) = ({T})((({W}){vr_cast(T)}(vs1, _li) {inst['op']} ({W}){vr_cast(T)}(vs2, _li) + 1) >> 1);
                PC(_li) += 4;
            }}
            NEXT();
        }}
        """)

    def tmpl_avg_vx(self, inst, sew):
        T = self.ctype(sew, inst["itype"])
        W = self.ctype(sew * 2, inst["itype"])
        return dedent(f"""\
        rvv_{inst['name']}_vx_e{sew}:
        {{
            int vd = ip[-1].rd, vs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
            {T} sc = *({T} *)&GPR(rs2, 0);
            uint32_t _vm = (ip[-1].inst >> 25) & 1;
            FOR_EACH_LANE
            {{
                if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1)))
                    {vr_cast(T)}(vd, _li) = ({T})((({W}){vr_cast(T)}(vs1, _li) {inst['op']} ({W})sc + 1) >> 1);
                PC(_li) += 4;
            }}
            NEXT();
        }}
        """)

    # ========================================================================
    # §8 整数宽化 (dst width = 2×SEW)
    # ========================================================================

    def tmpl_widen_vv(self, inst, sew):
        T = self.ctype(sew, inst["itype"])
        DT = self.ctype(sew * 2, inst["itype"])
        return dedent(f"""\
        rvv_{inst['name']}_vv_e{sew}:
        {{
            int vd = ip[-1].rd, vs1 = ip[-1].rs1, vs2 = ip[-1].rs2;
            uint32_t _vm = (ip[-1].inst >> 25) & 1;
            FOR_EACH_LANE
            {{
                if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1)))
                    {vr_cast(DT)}(vd, _li) = ({DT}){vr_cast(T)}(vs1, _li) {inst['op']} ({DT}){vr_cast(T)}(vs2, _li);
                PC(_li) += 4;
            }}
            NEXT();
        }}
        """)

    def tmpl_widen_vx(self, inst, sew):
        T = self.ctype(sew, inst["itype"])
        DT = self.ctype(sew * 2, inst["itype"])
        return dedent(f"""\
        rvv_{inst['name']}_vx_e{sew}:
        {{
            int vd = ip[-1].rd, vs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
            {T} sc = *({T} *)&GPR(rs2, 0);
            uint32_t _vm = (ip[-1].inst >> 25) & 1;
            FOR_EACH_LANE
            {{
                if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1)))
                    {vr_cast(DT)}(vd, _li) = ({DT}){vr_cast(T)}(vs1, _li) {inst['op']} ({DT})sc;
                PC(_li) += 4;
            }}
            NEXT();
        }}
        """)

    # ========================================================================
    # §9 整数窄化 (dst width = SEW/2)
    # ========================================================================

    def tmpl_narrow_vv(self, inst, sew):
        T = self.ctype(sew, inst["itype"])
        DT = self.ctype(sew // 2, inst["itype"])
        mask = (sew // 2) - 1
        return dedent(f"""\
        rvv_{inst['name']}_vv_e{sew}:
        {{
            int vd = ip[-1].rd, vs1 = ip[-1].rs1, vs2 = ip[-1].rs2;
            uint32_t _vm = (ip[-1].inst >> 25) & 1;
            FOR_EACH_LANE
            {{
                if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1)))
                    {vr_cast(DT)}(vd, _li) = ({DT})({vr_cast(T)}(vs1, _li) {inst['op']} ({vr_cast(T)}(vs2, _li) & {mask}U));
                PC(_li) += 4;
            }}
            NEXT();
        }}
        """)

    def tmpl_narrow_vx(self, inst, sew):
        T = self.ctype(sew, inst["itype"])
        DT = self.ctype(sew // 2, inst["itype"])
        mask = (sew // 2) - 1
        return dedent(f"""\
        rvv_{inst['name']}_vx_e{sew}:
        {{
            int vd = ip[-1].rd, vs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
            {T} sc = *({T} *)&GPR(rs2, 0) & {mask}U;
            uint32_t _vm = (ip[-1].inst >> 25) & 1;
            FOR_EACH_LANE
            {{
                if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1)))
                    {vr_cast(DT)}(vd, _li) = ({DT})({vr_cast(T)}(vs1, _li) {inst['op']} sc);
                PC(_li) += 4;
            }}
            NEXT();
        }}
        """)

    def tmpl_narrow_vi(self, inst, sew):
        T = self.ctype(sew, inst["itype"])
        DT = self.ctype(sew // 2, inst["itype"])
        mask = (sew // 2) - 1
        return dedent(f"""\
        rvv_{inst['name']}_vi_e{sew}:
        {{
            int vd = ip[-1].rd, vs1 = ip[-1].rs1;
            {T} imm = ({T})((int32_t)ip[-1].imm) & {mask}U;
            uint32_t _vm = (ip[-1].inst >> 25) & 1;
            FOR_EACH_LANE
            {{
                if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1)))
                    {vr_cast(DT)}(vd, _li) = ({DT})({vr_cast(T)}(vs1, _li) {inst['op']} imm);
                PC(_li) += 4;
            }}
            NEXT();
        }}
        """)

    # ========================================================================
    # §10 整数符号扩展
    # ========================================================================

    def tmpl_ext_v(self, inst, sew):
        T = self.ctype(sew, inst["itype"])
        factor = inst["factor"]
        DT = self.ctype(sew * factor, inst["itype"])
        return dedent(f"""\
        rvv_{inst['name']}_v_e{sew}:
        {{
            int vd = ip[-1].rd, vs1 = ip[-1].rs1;
            uint32_t _vm = (ip[-1].inst >> 25) & 1;
            FOR_EACH_LANE
            {{
                if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1)))
                    {vr_cast(DT)}(vd, _li) = ({DT}){vr_cast(T)}(vs1, _li);
                PC(_li) += 4;
            }}
            NEXT();
        }}
        """)

    # ========================================================================
    # §11 进位
    # ========================================================================

    def tmpl_carry_vv(self, inst, sew):
        T = self.ctype(sew, inst["itype"])
        W = self.ctype(sew * 2, inst["itype"])
        return dedent(f"""\
        rvv_{inst['name']}_vv_e{sew}:
        {{
            int vd = ip[-1].rd, vs1 = ip[-1].rs1, vs2 = ip[-1].rs2;
            uint32_t _vm = (ip[-1].inst >> 25) & 1;
            FOR_EACH_LANE
            {{
                if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1))) {{
                    {W} r = ({W}){vr_cast(T)}(vs1, _li) {inst['op']} ({W}){vr_cast(T)}(vs2, _li);
                    {vr_cast(T)}(vd, _li) = (r >> {sew}) & 1;
                }}
                PC(_li) += 4;
            }}
            NEXT();
        }}
        """)

    def tmpl_carry_vx(self, inst, sew):
        T = self.ctype(sew, inst["itype"])
        W = self.ctype(sew * 2, inst["itype"])
        return dedent(f"""\
        rvv_{inst['name']}_vx_e{sew}:
        {{
            int vd = ip[-1].rd, vs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
            {T} sc = *({T} *)&GPR(rs2, 0);
            uint32_t _vm = (ip[-1].inst >> 25) & 1;
            FOR_EACH_LANE
            {{
                if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1))) {{
                    {W} r = ({W}){vr_cast(T)}(vs1, _li) {inst['op']} ({W})sc;
                    {vr_cast(T)}(vd, _li) = (r >> {sew}) & 1;
                }}
                PC(_li) += 4;
            }}
            NEXT();
        }}
        """)

    def tmpl_carry_vi(self, inst, sew):
        T = self.ctype(sew, inst["itype"])
        W = self.ctype(sew * 2, inst["itype"])
        return dedent(f"""\
        rvv_{inst['name']}_vi_e{sew}:
        {{
            int vd = ip[-1].rd, vs1 = ip[-1].rs1;
            {T} imm = ({T})(int32_t)ip[-1].imm;
            uint32_t _vm = (ip[-1].inst >> 25) & 1;
            FOR_EACH_LANE
            {{
                if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1))) {{
                    {W} r = ({W}){vr_cast(T)}(vs1, _li) {inst['op']} ({W})imm;
                    {vr_cast(T)}(vd, _li) = (r >> {sew}) & 1;
                }}
                PC(_li) += 4;
            }}
            NEXT();
        }}
        """)

    # ========================================================================
    # §12-13 浮点二元/一元
    # ========================================================================

    def tmpl_fbin_vv(self, inst, sew):
        T = self.ctype(sew, inst["itype"])
        return dedent(f"""\
        rvv_{inst['name']}_vv_e{sew}:
        {{
            int vd = ip[-1].rd, vs1 = ip[-1].rs1, vs2 = ip[-1].rs2;
            uint32_t _vm = (ip[-1].inst >> 25) & 1;
            FOR_EACH_LANE
            {{
                if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1)))
                    {vr_cast(T)}(vd, _li) = {vr_cast(T)}(vs1, _li) {inst['op']} {vr_cast(T)}(vs2, _li);
                PC(_li) += 4;
            }}
            NEXT();
        }}
        """)

    def tmpl_fbin_vf(self, inst, sew):
        T = self.ctype(sew, inst["itype"])
        return dedent(f"""\
        rvv_{inst['name']}_vf_e{sew}:
        {{
            int vd = ip[-1].rd, vs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
            {T} sc = FR(rs2, 0);
            uint32_t _vm = (ip[-1].inst >> 25) & 1;
            FOR_EACH_LANE
            {{
                if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1)))
                    {vr_cast(T)}(vd, _li) = {vr_cast(T)}(vs1, _li) {inst['op']} sc;
                PC(_li) += 4;
            }}
            NEXT();
        }}
        """)

    def tmpl_funary_v(self, inst, sew):
        T = self.ctype(sew, inst["itype"])
        return dedent(f"""\
        rvv_{inst['name']}_v_e{sew}:
        {{
            int vd = ip[-1].rd, vs1 = ip[-1].rs1;
            uint32_t _vm = (ip[-1].inst >> 25) & 1;
            FOR_EACH_LANE
            {{
                if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1))) {{
                    {T} v = {vr_cast(T)}(vs1, _li);
                    {vr_cast(T)}(vd, _li) = ({T})({inst['expr']});
                }}
                PC(_li) += 4;
            }}
            NEXT();
        }}
        """)

    # ========================================================================
    # §15 浮点符号注入
    # ========================================================================

    def tmpl_fsgnj_vv(self, inst, sew):
        T = self.ctype(sew, inst["itype"])
        UT = self.ctype(sew, "uint")
        sign_bit = "31" if sew == 32 else "63"
        op = inst["op"]
        if op == "j":
            body = f"({vr_cast(UT)}(rs1, _li) & ~(1U << {sign_bit})) | ({vr_cast(UT)}(rs2, _li) & (1U << {sign_bit}))"
        elif op == "jn":
            body = f"({vr_cast(UT)}(rs1, _li) & ~(1U << {sign_bit})) | ((~{vr_cast(UT)}(rs2, _li)) & (1U << {sign_bit}))"
        else:
            body = f"{vr_cast(UT)}(rs1, _li) ^ ({vr_cast(UT)}(rs2, _li) & (1U << {sign_bit}))"
        return dedent(f"""\
        rvv_{inst['name']}_vv_e{sew}:
        {{
            int vd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
            uint32_t _vm = (ip[-1].inst >> 25) & 1;
            FOR_EACH_LANE
            {{
                if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1)))
                    {vr_cast(UT)}(vd, _li) = {body};
                PC(_li) += 4;
            }}
            NEXT();
        }}
        """)

    def tmpl_fsgnj_vf(self, inst, sew):
        UT = self.ctype(sew, "uint")
        sign_bit = "31" if sew == 32 else "63"
        op = inst["op"]
        if op == "j":
            body = f"({vr_cast(UT)}(rs1, _li) & ~(1U << {sign_bit})) | (scr & (1U << {sign_bit}))"
        elif op == "jn":
            body = f"({vr_cast(UT)}(rs1, _li) & ~(1U << {sign_bit})) | ((~scr) & (1U << {sign_bit}))"
        else:
            body = f"{vr_cast(UT)}(rs1, _li) ^ (scr & (1U << {sign_bit}))"
        return dedent(f"""\
        rvv_{inst['name']}_vf_e{sew}:
        {{
            int vd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
            {UT} scr = *({UT} *)&FR(rs2, 0);
            uint32_t _vm = (ip[-1].inst >> 25) & 1;
            FOR_EACH_LANE
            {{
                if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1)))
                    {vr_cast(UT)}(vd, _li) = {body};
                PC(_li) += 4;
            }}
            NEXT();
        }}
        """)

    # ========================================================================
    # §16 浮点分类
    # ========================================================================

    def tmpl_fclass_v(self, inst, sew):
        return dedent(f"""\
        rvv_{inst['name']}_v_e{sew}:
        {{
            int vd = ip[-1].rd, rs1 = ip[-1].rs1;
            uint32_t _vm = (ip[-1].inst >> 25) & 1;
            FOR_EACH_LANE
            {{
                if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1))) {{
                    uint32_t b = {vr_cast("uint32_t")}(rs1, _li);
                    uint32_t exp = (b >> 23) & 0xFF, m = b & 0x7FFFFF, sgn = (b >> 31) & 1;
                    int r = 0;
                    if (exp == 0xFF)
                        r = m ? (sgn ? (1<<9) : (1<<8)) : (sgn ? (1<<0) : (1<<7));
                    else if (exp == 0)
                        r = m ? (sgn ? (1<<2) : (1<<5)) : (sgn ? (1<<3) : (1<<4));
                    else
                        r = sgn ? (1<<1) : (1<<6);
                    {vr_cast("uint32_t")}(vd, _li) = (uint32_t)r;
                }}
                PC(_li) += 4;
            }}
            NEXT();
        }}
        """)

    # ========================================================================
    # §17 浮点 FMA
    # ========================================================================

    def tmpl_fma_vv(self, inst, sew):
        T = self.ctype(sew, inst["itype"])
        return dedent(f"""\
        rvv_{inst['name']}_vv_e{sew}:
        {{
            int vd = ip[-1].rd, vs1 = ip[-1].rs1, vs2 = ip[-1].rs2;
            uint32_t _vm = (ip[-1].inst >> 25) & 1;
            FOR_EACH_LANE
            {{
                if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1))) {{
                    {T} a = {vr_cast(T)}(vs1, _li), b = {vr_cast(T)}(vs2, _li), d = {vr_cast(T)}(vd, _li);
                    {vr_cast(T)}(vd, _li) = ({T})({inst['expr']});
                }}
                PC(_li) += 4;
            }}
            NEXT();
        }}
        """)

    def tmpl_fma_vf(self, inst, sew):
        T = self.ctype(sew, inst["itype"])
        return dedent(f"""\
        rvv_{inst['name']}_vf_e{sew}:
        {{
            int vd = ip[-1].rd, vs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
            {T} sc = FR(rs2, 0);
            uint32_t _vm = (ip[-1].inst >> 25) & 1;
            FOR_EACH_LANE
            {{
                if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1))) {{
                    {T} a = {vr_cast(T)}(vs1, _li), b = sc, d = {vr_cast(T)}(vd, _li);
                    {vr_cast(T)}(vd, _li) = ({T})({inst['expr']});
                }}
                PC(_li) += 4;
            }}
            NEXT();
        }}
        """)

    # ========================================================================
    # §18 浮点宽化
    # ========================================================================

    def tmpl_fwiden_vv(self, inst, sew):
        T = self.ctype(sew, inst["itype"])
        return dedent(f"""\
        rvv_{inst['name']}_vv_e{sew}:
        {{
            int vd = ip[-1].rd, vs1 = ip[-1].rs1, vs2 = ip[-1].rs2;
            uint32_t _vm = (ip[-1].inst >> 25) & 1;
            FOR_EACH_LANE
            {{
                if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1)))
                    {vr_cast("double")}(vd, _li) = (double){vr_cast(T)}(vs1, _li) {inst['op']} (double){vr_cast(T)}(vs2, _li);
                PC(_li) += 4;
            }}
            NEXT();
        }}
        """)

    def tmpl_fwiden_vf(self, inst, sew):
        T = self.ctype(sew, inst["itype"])
        return dedent(f"""\
        rvv_{inst['name']}_vf_e{sew}:
        {{
            int vd = ip[-1].rd, vs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
            {T} sc = FR(rs2, 0);
            uint32_t _vm = (ip[-1].inst >> 25) & 1;
            FOR_EACH_LANE
            {{
                if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1)))
                    {vr_cast("double")}(vd, _li) = (double){vr_cast(T)}(vs1, _li) {inst['op']} (double)sc;
                PC(_li) += 4;
            }}
            NEXT();
        }}
        """)

    # ========================================================================
    # §19 浮点宽化 FMA
    # ========================================================================

    def tmpl_fwiden_fma_vv(self, inst, sew):
        T = self.ctype(sew, inst["itype"])
        return dedent(f"""\
        rvv_{inst['name']}_vv_e{sew}:
        {{
            int vd = ip[-1].rd, vs1 = ip[-1].rs1, vs2 = ip[-1].rs2;
            uint32_t _vm = (ip[-1].inst >> 25) & 1;
            FOR_EACH_LANE
            {{
                if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1))) {{
                    double a = (double){vr_cast(T)}(vs1, _li), b = (double){vr_cast(T)}(vs2, _li);
                    double d = {vr_cast("double")}(vd, _li);
                    {vr_cast("double")}(vd, _li) = (double)({inst['expr']});
                }}
                PC(_li) += 4;
            }}
            NEXT();
        }}
        """)

    def tmpl_fwiden_fma_vf(self, inst, sew):
        T = self.ctype(sew, inst["itype"])
        return dedent(f"""\
        rvv_{inst['name']}_vf_e{sew}:
        {{
            int vd = ip[-1].rd, vs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
            {T} sc = FR(rs2, 0);
            uint32_t _vm = (ip[-1].inst >> 25) & 1;
            FOR_EACH_LANE
            {{
                if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1))) {{
                    double a = (double){vr_cast(T)}(vs1, _li), b = (double)sc;
                    double d = {vr_cast("double")}(vd, _li);
                    {vr_cast("double")}(vd, _li) = (double)({inst['expr']});
                }}
                PC(_li) += 4;
            }}
            NEXT();
        }}
        """)

    # ========================================================================
    # §20 浮点窄化
    # ========================================================================

    def tmpl_fnarrow_v(self, inst, sew):
        return dedent(f"""\
        rvv_{inst['name']}_v_e{sew}:
        {{
            int vd = ip[-1].rd, vs1 = ip[-1].rs1;
            uint32_t _vm = (ip[-1].inst >> 25) & 1;
            FOR_EACH_LANE
            {{
                if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1)))
                    {vr_cast("float")}(vd, _li) = (float){vr_cast("double")}(vs1, _li);
                PC(_li) += 4;
            }}
            NEXT();
        }}
        """)

    # ========================================================================
    # §21-22 归约
    # ========================================================================

    def _reduce_init(self, sew, itype, init_name):
        """将通用 init 名字转换为 SEW 特化的 limit 宏"""
        if init_name == "INT_MIN":
            return f"INT{sew}_MIN"
        elif init_name == "INT_MAX":
            return f"INT{sew}_MAX"
        elif init_name == "UINT_MAX":
            return f"UINT{sew}_MAX"
        elif init_name == "~0":
            return f"(uint{sew}_t)(-1)"
        else:
            return init_name

    def tmpl_reduce_vs(self, inst, sew):
        T = self.ctype(sew, inst["itype"])
        init = self._reduce_init(sew, inst["itype"], inst["init"])
        op = inst["op"]
        if op in ("+", "&", "|", "^"):
            acc_stmt = f"acc {op}= {vr_cast(T)}(vs1, _li)"
        elif op == "max":
            acc_stmt = f"if ({vr_cast(T)}(vs1, _li) > acc) acc = {vr_cast(T)}(vs1, _li)"
        elif op == "min":
            acc_stmt = f"if ({vr_cast(T)}(vs1, _li) < acc) acc = {vr_cast(T)}(vs1, _li)"
        else:
            acc_stmt = f"acc {op}= {vr_cast(T)}(vs1, _li)"
        return dedent(f"""\
        rvv_{inst['name']}_vs_e{sew}:
        {{
            int vd = ip[-1].rd, vs1 = ip[-1].rs1;
            uint32_t _vm = (ip[-1].inst >> 25) & 1;
            {T} acc = ({T})({init});
            FOR_EACH_LANE
            {{
                if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1)))
                    {acc_stmt};
                PC(_li) += 4;
            }}
            {vr_cast(T)}(vd, 0) = acc;
            NEXT();
        }}
        """)

    def tmpl_freduce_vs(self, inst, sew):
        T = self.ctype(sew, inst["itype"])
        init = inst["init"]
        op = inst["op"]
        if op == "+":
            acc = f"acc += {vr_cast(T)}(vs1, _li)"
        elif op == "max":
            acc = f"if ({vr_cast(T)}(vs1, _li) > acc || isnan(acc)) acc = {vr_cast(T)}(vs1, _li)"
        elif op == "min":
            acc = f"if ({vr_cast(T)}(vs1, _li) < acc || isnan(acc)) acc = {vr_cast(T)}(vs1, _li)"
        else:
            acc = f"acc {op}= {vr_cast(T)}(vs1, _li)"
        return dedent(f"""\
        rvv_{inst['name']}_vs_e{sew}:
        {{
            int vd = ip[-1].rd, vs1 = ip[-1].rs1;
            uint32_t _vm = (ip[-1].inst >> 25) & 1;
            {T} acc = ({T})({init});
            FOR_EACH_LANE
            {{
                if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1)))
                    {acc};
                PC(_li) += 4;
            }}
            {vr_cast(T)}(vd, 0) = acc;
            NEXT();
        }}
        """)

    # ========================================================================
    # §23-24 掩码 (merge 在 §23, mask_ops 用 bin 模板需要专门处理)
    # ========================================================================
    # mask_ops uses non-standard ops — need a special template

    def tmpl_mask_bin_vv(self, inst, sew):
        """掩码逻辑运算 — 通过 expr 表达式定义, a=vs1[i], b=vs2[i]"""
        T = self.ctype(sew, inst["itype"])
        expr = inst["expr"]
        return dedent(f"""\
        rvv_{inst['name']}_vv_e{sew}:
        {{
            int vd = ip[-1].rd, vs1 = ip[-1].rs1, vs2 = ip[-1].rs2;
            FOR_EACH_LANE
            {{
                if ((uint32_t)_li < _vl) {{
                    {T} a = {vr_cast(T)}(vs1, _li), b = {vr_cast(T)}(vs2, _li);
                    {vr_cast(T)}(vd, _li) = ({T})({expr});
                }}
                PC(_li) += 4;
            }}
            NEXT();
        }}
        """)

    # ========================================================================
    # §25 掩码计数
    # ========================================================================

    def tmpl_mcount_v(self, inst, sew):
        T = self.ctype(sew, inst["itype"])
        op = inst["op"]
        if op == "cpop":
            body = f"{vr_cast(T)}(vd, _li) = ({T})__builtin_popcount({vr_cast(T)}(vs1, _li))"
            decl = "int vd = ip[-1].rd, vs1 = ip[-1].rs1;"
        elif op == "first":
            body = f"""{vr_cast(T)}(vd, _li) = ({T})({vr_cast(T)}(vs1, _li) ? __builtin_ctz({vr_cast(T)}(vs1, _li)) : -1)"""
            decl = "int vd = ip[-1].rd, vs1 = ip[-1].rs1;"
        elif op == "id":
            body = f"{vr_cast(T)}(vd, _li) = ({T})_li"
            decl = f"int vd = ip[-1].rd; (void)ip[-1].rs1;"
        else:
            body = f"{vr_cast(T)}(vd, _li) = ({T})_li /* TODO: {op} */"
            decl = f"int vd = ip[-1].rd; (void)ip[-1].rs1;"
        return dedent(f"""\
        rvv_{inst['name']}_v_e{sew}:
        {{
            {decl}
            FOR_EACH_LANE
            {{
                if ((uint32_t)_li < _vl) {{
                    {body};
                }}
                PC(_li) += 4;
            }}
            NEXT();
        }}
        """)

    # ========================================================================
    # §26-30 访存
    # ========================================================================

    def _elem_bytes(self, sew):
        return sew // 8

    def tmpl_ldst_v(self, inst, sew):
        UT = self.ctype(sew, "uint")
        eb = self._elem_bytes(sew)
        if inst["dir"] == "ld":
            return dedent(f"""\
            rvv_{inst['name']}_v_e{sew}:
            {{
                int vd = ip[-1].rd, rs1 = ip[-1].rs1;
                uint32_t _vm = (ip[-1].inst >> 25) & 1;
                FOR_EACH_LANE
                {{
                    if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1))) {{
                        uint32_t a = GPR(rs1, _li) + _li * {eb};
                        {vr_cast(UT)}(vd, _li) = (a + {eb} <= s->vram_size) ? *({UT} *)(s->vram_ptr + a) : 0;
                    }}
                    PC(_li) += 4;
                }}
                NEXT();
            }}
            """)
        else:
            return dedent(f"""\
            rvv_{inst['name']}_v_e{sew}:
            {{
                int vs3 = ip[-1].rd, rs1 = ip[-1].rs1;
                uint32_t _vm = (ip[-1].inst >> 25) & 1;
                FOR_EACH_LANE
                {{
                    if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1))) {{
                        uint32_t a = GPR(rs1, _li) + _li * {eb};
                        if (a + {eb} <= s->vram_size) *({UT} *)(s->vram_ptr + a) = {vr_cast(UT)}(vs3, _li);
                    }}
                    PC(_li) += 4;
                }}
                NEXT();
            }}
            """)

    def tmpl_strided_v(self, inst, sew):
        UT = self.ctype(sew, "uint")
        eb = self._elem_bytes(sew)
        if inst["dir"] == "ld":
            return dedent(f"""\
            rvv_{inst['name']}_v_e{sew}:
            {{
                int vd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
                int32_t stride = (int32_t)GPR(rs2, 0);
                uint32_t _vm = (ip[-1].inst >> 25) & 1;
                FOR_EACH_LANE
                {{
                    if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1))) {{
                        uint32_t a = GPR(rs1, _li) + _li * stride;
                        {vr_cast(UT)}(vd, _li) = (a + {eb} <= s->vram_size) ? *({UT} *)(s->vram_ptr + a) : 0;
                    }}
                    PC(_li) += 4;
                }}
                NEXT();
            }}
            """)
        else:
            return dedent(f"""\
            rvv_{inst['name']}_v_e{sew}:
            {{
                int vs3 = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
                int32_t stride = (int32_t)GPR(rs2, 0);
                uint32_t _vm = (ip[-1].inst >> 25) & 1;
                FOR_EACH_LANE
                {{
                    if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1))) {{
                        uint32_t a = GPR(rs1, _li) + _li * stride;
                        if (a + {eb} <= s->vram_size) *({UT} *)(s->vram_ptr + a) = {vr_cast(UT)}(vs3, _li);
                    }}
                    PC(_li) += 4;
                }}
                NEXT();
            }}
            """)

    def tmpl_indexed_vv(self, inst, sew):
        UT = self.ctype(sew, "uint")
        eb = self._elem_bytes(sew)
        if inst["dir"] == "ld":
            return dedent(f"""\
            rvv_{inst['name']}_vv_e{sew}:
            {{
                int vd = ip[-1].rd, rs1 = ip[-1].rs1, vs2 = ip[-1].rs2;
                uint32_t _vm = (ip[-1].inst >> 25) & 1;
                FOR_EACH_LANE
                {{
                    if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1))) {{
                        uint32_t a = GPR(rs1, _li) + {vr_cast(UT)}(vs2, _li);
                        {vr_cast(UT)}(vd, _li) = (a + {eb} <= s->vram_size) ? *({UT} *)(s->vram_ptr + a) : 0;
                    }}
                    PC(_li) += 4;
                }}
                NEXT();
            }}
            """)
        else:
            return dedent(f"""\
            rvv_{inst['name']}_vv_e{sew}:
            {{
                int vs3 = ip[-1].rd, rs1 = ip[-1].rs1, vs2 = ip[-1].rs2;
                uint32_t _vm = (ip[-1].inst >> 25) & 1;
                FOR_EACH_LANE
                {{
                    if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1))) {{
                        uint32_t a = GPR(rs1, _li) + {vr_cast(UT)}(vs2, _li);
                        if (a + {eb} <= s->vram_size) *({UT} *)(s->vram_ptr + a) = {vr_cast(UT)}(vs3, _li);
                    }}
                    PC(_li) += 4;
                }}
                NEXT();
            }}
            """)

    def tmpl_wreg_v(self, inst, sew):
        """全寄存器加载/存储 — 所有 32 lanes 无条件操作"""
        UT = self.ctype(sew, "uint")
        eb = sew // 8
        if inst["dir"] == "ld":
            return dedent(f"""\
            rvv_{inst['name']}_v_e{sew}:
            {{
                int vd = ip[-1].rd, rs1 = ip[-1].rs1;
                for (int _li = 0; _li < 32; _li++) {{
                    uint32_t a = GPR(rs1, _li) + _li * {eb};
                    {vr_cast(UT)}(vd, _li) = (a + {eb} <= s->vram_size) ? *({UT} *)(s->vram_ptr + a) : 0;
                    PC(_li) += 4;
                }}
                NEXT();
            }}
            """)
        else:
            return dedent(f"""\
            rvv_{inst['name']}_v_e{sew}:
            {{
                int vs3 = ip[-1].rd, rs1 = ip[-1].rs1;
                for (int _li = 0; _li < 32; _li++) {{
                    uint32_t a = GPR(rs1, _li) + _li * {eb};
                    if (a + {eb} <= s->vram_size) *({UT} *)(s->vram_ptr + a) = {vr_cast(UT)}(vs3, _li);
                    PC(_li) += 4;
                }}
                NEXT();
            }}
            """)

    def tmpl_ff_v(self, inst, sew):
        """Fault-only-first: 类似 unit-stride load, 但在首个 fault 处停止并更新 vl"""
        UT = self.ctype(sew, "uint")
        eb = sew // 8
        return dedent(f"""\
        rvv_{inst['name']}_v_e{sew}:
        {{
            int vd = ip[-1].rd, rs1 = ip[-1].rs1;
            uint32_t _vm = (ip[-1].inst >> 25) & 1;
            int _ff_vl = 0;
            FOR_EACH_LANE
            {{
                if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1))) {{
                    uint32_t a = GPR(rs1, _li) + _li * {eb};
                    if (a + {eb} <= s->vram_size)
                        {vr_cast(UT)}(vd, _li) = *({UT} *)(s->vram_ptr + a);
                    else {{
                        _ff_vl = _li;
                        break;
                    }}
                }}
                PC(_li) += 4;
            }}
            if (_ff_vl > 0) _vl = _ff_vl;
            NEXT();
        }}
        """)

    # ========================================================================
    # §31-32 排列与滑窗
    # ========================================================================

    def tmpl_perm_vv(self, inst, sew):
        T = self.ctype(sew, inst["itype"])
        if inst["op"] == "gather":
            return dedent(f"""\
            rvv_{inst['name']}_vv_e{sew}:
            {{
                int vd = ip[-1].rd, vs1 = ip[-1].rs1, vs2 = ip[-1].rs2;
                uint32_t _vm = (ip[-1].inst >> 25) & 1;
                FOR_EACH_LANE
                {{
                    if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1))) {{
                        uint32_t idx = {vr_cast("uint32_t")}(vs2, _li);
                        if (idx < 32) {vr_cast(T)}(vd, _li) = {vr_cast(T)}(vs1, idx);
                    }}
                    PC(_li) += 4;
                }}
                NEXT();
            }}
            """)
        else:  # compress — only write if mask bit is set
            return dedent(f"""\
            rvv_{inst['name']}_vv_e{sew}:
            {{
                int vd = ip[-1].rd, vs1 = ip[-1].rs1, vs2 = ip[-1].rs2;
                int _wi = 0;
                FOR_EACH_LANE
                {{
                    if ((uint32_t)_li < _vl && ({vr_cast("uint32_t")}(vs2, _li) & 1)) {{
                        {vr_cast(T)}(vd, _wi) = {vr_cast(T)}(vs1, _li);
                        _wi++;
                    }}
                    PC(_li) += 4;
                }}
                NEXT();
            }}
            """)

    def tmpl_perm_vx(self, inst, sew):
        T = self.ctype(sew, inst["itype"])
        return dedent(f"""\
        rvv_{inst['name']}_vx_e{sew}:
        {{
            int vd = ip[-1].rd, vs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
            uint32_t base = *({T} *)&GPR(rs2, 0);
            uint32_t _vm = (ip[-1].inst >> 25) & 1;
            FOR_EACH_LANE
            {{
                if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1))) {{
                    uint32_t idx = base + {vr_cast("uint32_t")}(vs1, _li);
                    if (idx < 32) {vr_cast(T)}(vd, _li) = {vr_cast(T)}(vs1, idx);
                }}
                PC(_li) += 4;
            }}
            NEXT();
        }}
        """)

    def tmpl_slide_vx(self, inst, sew):
        T = self.ctype(sew, inst["itype"])
        op = inst["op"]
        if op == "up":
            shift = f"*({T} *)&GPR(rs2, 0)"
            return dedent(f"""\
            rvv_{inst['name']}_vx_e{sew}:
            {{
                int vd = ip[-1].rd, vs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
                int32_t off = (int32_t){shift};
                uint32_t _vm = (ip[-1].inst >> 25) & 1;
                FOR_EACH_LANE
                {{
                    if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1))) {{
                        int src = (int)_li - off;
                        {vr_cast(T)}(vd, _li) = (src >= 0 && src < 32) ? {vr_cast(T)}(vs1, src) : 0;
                    }}
                    PC(_li) += 4;
                }}
                NEXT();
            }}
            """)
        elif op == "down":
            shift = f"*({T} *)&GPR(rs2, 0)"
            return dedent(f"""\
            rvv_{inst['name']}_vx_e{sew}:
            {{
                int vd = ip[-1].rd, vs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
                int32_t off = (int32_t){shift};
                uint32_t _vm = (ip[-1].inst >> 25) & 1;
                FOR_EACH_LANE
                {{
                    if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1))) {{
                        int src = (int)_li + off;
                        {vr_cast(T)}(vd, _li) = (src >= 0 && src < 32) ? {vr_cast(T)}(vs1, src) : 0;
                    }}
                    PC(_li) += 4;
                }}
                NEXT();
            }}
            """)
        else:  # slide1up/slide1down — shift by 1, fill with scalar from FPR
            if op == "1up":
                return dedent(f"""\
                rvv_{inst['name']}_vx_e{sew}:
                {{
                    int vd = ip[-1].rd, vs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
                    {T} fill = FR(rs2, 0);
                    uint32_t _vm = (ip[-1].inst >> 25) & 1;
                    FOR_EACH_LANE
                    {{
                        if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1)))
                            {vr_cast(T)}(vd, _li) = (_li == 0) ? fill : {vr_cast(T)}(vs1, _li - 1);
                        PC(_li) += 4;
                    }}
                    NEXT();
                }}
                """)
            else:
                return dedent(f"""\
                rvv_{inst['name']}_vx_e{sew}:
                {{
                    int vd = ip[-1].rd, vs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
                    {T} fill = FR(rs2, 0);
                    uint32_t _vm = (ip[-1].inst >> 25) & 1;
                    FOR_EACH_LANE
                    {{
                        if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1)))
                            {vr_cast(T)}(vd, _li) = (_li == 31) ? fill : {vr_cast(T)}(vs1, _li + 1);
                        PC(_li) += 4;
                    }}
                    NEXT();
                }}
                """)

    def tmpl_slide_vi(self, inst, sew):
        T = self.ctype(sew, inst["itype"])
        op = inst["op"]
        if op == "up":
            return dedent(f"""\
            rvv_{inst['name']}_vi_e{sew}:
            {{
                int vd = ip[-1].rd, vs1 = ip[-1].rs1;
                int32_t off = (int32_t)ip[-1].imm;
                uint32_t _vm = (ip[-1].inst >> 25) & 1;
                FOR_EACH_LANE
                {{
                    if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1))) {{
                        int src = (int)_li - off;
                        {vr_cast(T)}(vd, _li) = (src >= 0 && src < 32) ? {vr_cast(T)}(vs1, src) : 0;
                    }}
                    PC(_li) += 4;
                }}
                NEXT();
            }}
            """)
        else:
            return dedent(f"""\
            rvv_{inst['name']}_vi_e{sew}:
            {{
                int vd = ip[-1].rd, vs1 = ip[-1].rs1;
                int32_t off = (int32_t)ip[-1].imm;
                uint32_t _vm = (ip[-1].inst >> 25) & 1;
                FOR_EACH_LANE
                {{
                    if ((uint32_t)_li < _vl && (_vm || ({vr_cast("uint32_t")}(0, _li) & 1))) {{
                        int src = (int)_li + off;
                        {vr_cast(T)}(vd, _li) = (src >= 0 && src < 32) ? {vr_cast(T)}(vs1, src) : 0;
                    }}
                    PC(_li) += 4;
                }}
                NEXT();
            }}
            """)

    # ========================================================================
    # §33 配置
    # ========================================================================

    def tmpl_config_v(self, inst, sew):
        return dedent(f"""\
        rvv_{inst['name']}_v_e{sew}:
        {{
            int rd = ip[-1].rd, rs1 = ip[-1].rs1;
            uint32_t avl = GPR(rs1, 0);
            uint32_t new_vl = avl ? (avl < 32 ? avl : 32) : 32;
            if (vl) *vl = new_vl;
            _vl = new_vl;
            FOR_EACH_LANE
            {{
                GPR(rd, _li) = new_vl;
                PC(_li) += 4;
            }}
            NEXT();
        }}
        """)

    # ========================================================================
    # 生成主逻辑
    # ========================================================================

    def _header(self, family):
        forms = ", ".join(family.get("forms", []))
        n = len(family.get("instructions", []))
        return f"""\
/*
 * rvv_{family['name']}.h — {n} instructions ({forms})
 * Auto-generated by gen_rvv.py — DO NOT EDIT
 */
"""

    def generate(self):
        os.makedirs(OUT_DIR, exist_ok=True)
        for family in self.spec["families"]:
            self._gen_family(family)

    def _gen_family(self, family):
        tmpl_base = family["template"]
        default_forms = family.get("forms", ["vv"])
        sew_list = [s for s in family["sew"] if s in SUPPORTED_SEW]

        output = self._header(family)

        for inst in family["instructions"]:
            inst_forms = inst.get("forms", default_forms)
            sew_exclude = inst.get("sew_exclude", [])
            decode = inst.get("decode", {})
            dispatch_label = decode.get("label")  # e.g. "op_vadd_vv"

            # 空钩子 label — dispatch init 引用它, rebind 立即覆盖
            # 放在所有 SEW 实现之前, 不绑定任何具体 SEW
            if dispatch_label:
                output += f"{dispatch_label}:\n\n"

            for sew in sew_list:
                if sew in sew_exclude:
                    continue
                for form in inst_forms:
                    tmpl_name = f"tmpl_{tmpl_base}_{form}"
                    try:
                        tmpl_fn = getattr(self, tmpl_name)
                    except AttributeError:
                        print(f"  WARN: template {tmpl_name} not found, skipping "
                              f"{inst['name']}.{form}_e{sew}")
                        continue
                    output += tmpl_fn(inst, sew) + "\n"

        fname = os.path.join(OUT_DIR, f"rvv_{family['name']}.h")
        with open(fname, "w") as f:
            f.write(output)
        print(f"  GEN  {os.path.relpath(fname, SCRIPT_DIR)}")

    def _gen_all_include(self):
        lines = [
            "/* rvv_all.h — RVV 1.0 all handlers",
            " * Auto-generated by gen_rvv.py — DO NOT EDIT",
            " */",
            "",
        ]
        for family in self.spec["families"]:
            lines.append(f'#include "rvv_{family["name"]}.h"')
        # misc dispatch labels (vfmul_vf, vle32_v, vse32_v, ...) — 空钩子
        misc_labels = self._gen_misc_dispatch_labels()
        if misc_labels:
            lines.append("")
            lines.append("/* misc dispatch labels — handler 复用已有实现 */")
            lines.append(misc_labels)
        lines.append("")
        fname = os.path.join(OUT_DIR, "rvv_all.h")
        with open(fname, "w") as f:
            f.write("\n".join(lines))
        print(f"  GEN  {os.path.relpath(fname, SCRIPT_DIR)}")

    def _gen_misc_dispatch_labels(self):
        """为 misc 段条目生成空 dispatch 钩子 (handler 复用已有实现)"""
        misc = self.spec.get("misc", [])
        if not misc:
            return ""
        labels = []
        for entry in misc:
            decode = entry.get("decode", {})
            label = decode.get("label")
            if label:
                labels.append(f"{label}:")
                labels.append("    /* dispatch hook — rebind overwrites, see dispatch_rebind.h */")
                labels.append("")
        return "\n".join(labels)


if __name__ == "__main__":
    spec_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SPEC_DIR, "rvv_spec.yaml")
    gen = RVVGen(spec_path)
    gen.generate()
    gen._gen_all_include()
    total = sum(
        len(fam.get("instructions", [])) * len([s for s in fam.get("sew", []) if s in SUPPORTED_SEW]) * len(fam.get("forms", ["vv"]))
        for fam in gen.spec["families"]
    )
    print(f"\nDone. {len(gen.spec['families'])} families → generated/")
