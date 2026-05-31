#!/usr/bin/env python3
"""Generate Plan 9 WORD macros for AVX512-FP16 (EVEX MAP5 W0)."""

def enc_rr(opcode, rd, rn, rm, vl=0):
    p0 = ((~rd >> 3) & 1) << 7 | 1 << 6 | ((~rm >> 3) & 1) << 5 | ((rd >> 4) & 1) << 4 | 0b101
    p1 = ((~rn & 0xF) << 3) | 0b10000000
    p2 = (vl << 5)
    modrm = 0b11000000 | ((rd & 7) << 3) | (rm & 7)
    return bytes([0x62, p0 & 0xFF, p1 & 0xFF, p2 & 0xFF, opcode, modrm])


def enc_rm(opcode, rd, rn, base, vl=0):
    p0 = ((~rd >> 3) & 1) << 7 | 1 << 6 | 1 << 5 | ((rd >> 4) & 1) << 4 | 0b101
    p1 = ((~rn & 0xF) << 3) | 0b10000000
    p2 = (vl << 5)
    modrm = 0b00000100 | ((rd & 7) << 3)
    sib = (0 << 6) | (4 << 3) | (base & 7)
    return bytes([0x62, p0 & 0xFF, p1 & 0xFF, p2 & 0xFF, opcode, modrm, sib])


def word_line(name, blob):
    parts = "; ".join(f"WORD $0x{b:02x}" for b in blob)
    return f"#define {name} {parts}"


def reg(name):
    return int(name[1:])


def main():
    lines = [
        "// AVX512-FP16 WORD encodings (EVEX MAP5 W0, no mask).",
        "// AT&T: OP src2, src1, dest  =>  dest = src1 OP src2",
        "",
    ]

    bases = {"SI": 6, "DI": 7, "R8": 8, "DX": 2}
    for base_name, base in bases.items():
        for reg_name, rd, vl in [("Y0", 0, 1), ("Y1", 1, 1), ("X0", 0, 0), ("X1", 1, 0)]:
            tag = f"{reg_name[0]}{rd}" if reg_name[0] == 'Y' else reg_name
            lines.append(word_line(
                f"VMOVUPH_{tag}_{base_name}",
                enc_rm(0x6F, rd, rd, base, vl),
            ))

    # dest=0, rn=0, rm=1 => Y0 = Y0 op Y1
    for op_name, opcode in [
        ("VADDPH", 0x58), ("VSUBPH", 0x5C), ("VMULPH", 0x59),
        ("VDIVPH", 0x5E), ("VMAXPH", 0x62), ("VMINPH", 0x5D),
    ]:
        for vl, suf in [(0, "X0"), (1, "Y0")]:
            lines.append(word_line(
                f"{op_name}_{suf}_Y1_{suf}",
                enc_rr(opcode, 0, 0, 1, vl),
            ))

    # Y3 = Y1 * Y2
    lines.append(word_line("VMULPH_Y3_Y1_Y2", enc_rr(0x59, 3, 1, 2, 1)))
    lines.append(word_line("VMULPH_Y3_Y0_Y1", enc_rr(0x59, 3, 0, 1, 1)))
    lines.append(word_line("VMULPH_X3_X1_X2", enc_rr(0x59, 3, 1, 2, 0)))
    lines.append(word_line("VMULPH_X3_X0_X1", enc_rr(0x59, 3, 0, 1, 0)))
    lines.append(word_line("VMULPH_X4_X3_X2", enc_rr(0x59, 4, 3, 2, 0)))

    # sqrt: VSQRTPH X5, X4 -> X5 = sqrt(X4) => rd=5, rn=4, rm=4 for unary?
    # Intel VSQRTPH xmm1, xmm2/m -> xmm1 = sqrt(xmm2)
    # Go VSQRTPH X4, X5 -> X5 = sqrt(X4) => rd=5, rn=4, rm=4
    lines.append(word_line("VSQRTPH_X5_X4", enc_rr(0x51, 5, 4, 4, 0)))
    lines.append(word_line("VSQRTPH_Y5_Y4", enc_rr(0x51, 5, 4, 4, 1)))

    # Y10 = Y10 + Y0
    lines.append(word_line("VADDPH_Y10_Y0_Y10", enc_rr(0x58, 10, 10, 0, 1)))
    lines.append(word_line("VADDPH_X10_X1_X10", enc_rr(0x58, 10, 10, 1, 0)))
    lines.append(word_line("VMAXPH_Y10_Y0_Y10", enc_rr(0x62, 10, 10, 0, 1)))
    lines.append(word_line("VMAXPH_X10_X1_X10", enc_rr(0x62, 10, 10, 1, 0)))
    lines.append(word_line("VMINPH_Y10_Y0_Y10", enc_rr(0x5D, 10, 10, 0, 1)))
    lines.append(word_line("VMINPH_X10_X1_X10", enc_rr(0x5D, 10, 10, 1, 0)))

    # X1 = max(X1, X2)
    lines.append(word_line("VADDPH_X1_X2_X1", enc_rr(0x58, 1, 1, 2, 0)))
    lines.append(word_line("VADDPH_X0_X2_X0", enc_rr(0x58, 0, 0, 2, 0)))
    lines.append(word_line("VADDPH_X0_X4_X0", enc_rr(0x58, 0, 0, 4, 0)))
    lines.append(word_line("VMAXPH_X0_X2_X0", enc_rr(0x62, 0, 0, 2, 0)))
    lines.append(word_line("VMAXPH_X4_X2_X3", enc_rr(0x62, 4, 2, 3, 0)))
    lines.append(word_line("VADDPH_X4_X2_X3", enc_rr(0x58, 4, 2, 3, 0)))
    lines.append(word_line("VMAXPH_X0_X4_X7", enc_rr(0x62, 0, 4, 7, 0)))
    lines.append(word_line("VADDPH_X0_X5_X8", enc_rr(0x58, 0, 5, 8, 0)))
    lines.append(word_line("VMAXPH_X0_X5_X8", enc_rr(0x62, 0, 5, 8, 0)))
    lines.append(word_line("VMULPH_X0_X2_X0", enc_rr(0x59, 0, 0, 2, 0)))
    lines.append(word_line("VMULPH_X0_X1_X0", enc_rr(0x59, 0, 0, 1, 0)))
    lines.append(word_line("VMAXPH_X0_X1_X0", enc_rr(0x62, 0, 0, 1, 0)))
    lines.append(word_line("VMINPH_X0_X1_X0", enc_rr(0x5D, 0, 0, 1, 0)))
    lines.append(word_line("VMAXPH_X14_X0_X14", enc_rr(0x62, 14, 14, 0, 0)))
    lines.append(word_line("VMINPH_X14_X0_X14", enc_rr(0x5D, 14, 14, 0, 0)))
    lines.append(word_line("VADDPH_X5_X3_X4", enc_rr(0x58, 5, 3, 4, 0)))
    lines.append(word_line("VADDPH_X8_X6_X7", enc_rr(0x58, 8, 6, 7, 0)))
    lines.append(word_line("VMAXPH_X5_X3_X4", enc_rr(0x62, 5, 3, 4, 0)))
    lines.append(word_line("VMAXPH_X8_X6_X7", enc_rr(0x62, 8, 6, 7, 0)))
    lines.append(word_line("VMAXPH_X5_X13_X10", enc_rr(0x62, 10, 5, 13, 0)))
    lines.append(word_line("VMAXPH_Y5_Y13_Y10", enc_rr(0x62, 10, 5, 13, 1)))
    lines.append(word_line("VDIVPH_X0_X0_X3", enc_rr(0x5E, 0, 0, 3, 0)))
    lines.append(word_line("VDIVPH_X1_X0_X2", enc_rr(0x5E, 1, 0, 2, 0)))

    lines.append(word_line("VADDPH_Y0_Y3_Y0", enc_rr(0x58, 0, 0, 3, 1)))
    lines.append(word_line("VMULPH_Y3_Y1_Y14", enc_rr(0x59, 3, 1, 14, 1)))
    lines.append(word_line("VMULPH_X3_X1_X14", enc_rr(0x59, 3, 1, 14, 0)))
    lines.append(word_line("VADDPH_X0_X3_X0", enc_rr(0x58, 0, 0, 3, 0)))
    lines.append(word_line("VPBROADCASTW_X11_X10", enc_rr(0x7A, 11, 10, 10, 0)))
    lines.append(word_line("VPBROADCASTW_Y14_X14", enc_rr(0x7A, 14, 14, 14, 1)))

    # PhaseCoupling register patterns (AT&T).
    coupling = [
        ("VMULPH_X3_X2_X4", 0x59, 3, 2, 4, 0),
        ("VMULPH_X1_X0_X7", 0x59, 1, 0, 7, 0),
        ("VMULPH_X5_X5_X6", 0x59, 5, 5, 6, 0),
        ("VDIVPH_X6_X7_X8", 0x5E, 6, 7, 8, 0),
        ("VMULPH_Y3_Y2_Y4", 0x59, 3, 2, 4, 1),
        ("VMULPH_Y1_Y0_Y7", 0x59, 1, 0, 7, 1),
        ("VMULPH_Y5_Y5_Y6", 0x59, 5, 5, 6, 1),
        ("VDIVPH_Y6_Y7_Y8", 0x5E, 6, 7, 8, 1),
    ]
    for name, opc, rd, rn, rm, vl in coupling:
        lines.append(word_line(name, enc_rr(opc, rd, rn, rm, vl)))

    for op_name, opcode in [
        ("VADDPH", 0x58), ("VSUBPH", 0x5C), ("VMULPH", 0x59),
        ("VDIVPH", 0x5E), ("VMAXPH", 0x62), ("VMINPH", 0x5D),
    ]:
        lines.append(word_line(
            f"{op_name}_X2_X3_X2",
            enc_rr(opcode, 2, 2, 3, 0),
        ))

    lines.append(word_line("VMAXPH_X0_X31_X0", enc_rr(0x62, 0, 0, 31, 0)))
    lines.append(word_line("VMAXPH_X2_X31_X2", enc_rr(0x62, 2, 2, 31, 0)))
    lines.append(word_line("VMAXPH_Y0_Y31_Y0", enc_rr(0x62, 0, 0, 31, 1)))
    lines.append(word_line("VSQRTPH_X0_X0", enc_rr(0x51, 0, 0, 0, 0)))
    lines.append(word_line("VSQRTPH_Y0_Y0", enc_rr(0x51, 0, 0, 0, 1)))

    # store ymm0 to (DI): use VMOVDQU after extract - keep as comments
    lines.append("")
    lines.append("#define STORE_Y0_16H_DI \\")
    lines.append("\tVEXTRACTI128 $0, Y0, X2; \\")
    lines.append("\tVEXTRACTI128 $1, Y0, X3; \\")
    lines.append("\tVMOVDQU X2, (DI); \\")
    lines.append("\tVMOVDQU X3, 16(DI)")
    lines.append("")
    lines.append("#define STORE_X0_8H_DI \\")
    lines.append("\tVMOVDQU X0, (DI)")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
