// NEON bf16 lane helpers: VZIP bit-cast into .S4, VUZP2 pack, RNE round.

#define BF16_BITS_TO_F32_LOW(src_h8, dst_h8)  VZIP1 src_h8, V31.H8, dst_h8
#define BF16_BITS_TO_F32_HIGH(src_h8, dst_h8) VZIP2 src_h8, V31.H8, dst_h8
#define F32_BITS_TO_BF16_H8(s0_h8, s1_h8, dst_h8) VUZP2 s1_h8, s0_h8, dst_h8

#define VMOV_VREG(src, dst) WORD $(0x4EA01C00 | ((src) << 16) | ((src) << 5) | (dst))

#define F32_BITS_TO_BF16_H4(src_s4) \
    VMOV_VREG(src_s4, 8); \
    VMOV_VREG(src_s4, 9); \
    VUZP2 V9.H8, V8.H8, V12.H8

#define BF16_BITS_TO_F32_H4(src_h8, dst_s4) \
    VZIP1 src_h8, V31.H8, dst_s4

#define ROUND_BF16_S4(reg_s4)             \
    VUSHR $16, reg_s4, V16.S4             \
    VAND  V29.B16, V16.B16, V16.B16       \
    VADD  V28.S4, V16.S4, V16.S4          \
    VADD  V16.S4, reg_s4, reg_s4

#define BF16_RNE_BITS_IN_REG(reg) \
    MOVD $0x7fff, R8; \
    LSR $16, reg, R9; \
    AND $1, R9, R9; \
    ADD R8, R9, R9; \
    ADD R9, reg, reg; \
    LSR $16, reg, reg

#define BF16_RNE_INIT \
    VEOR V31.B16, V31.B16, V31.B16; \
    MOVD $0x7fff, R6; \
    VDUP R6, V28.S4; \
    MOVD $1, R6; \
    VDUP R6, V29.S4

#define BF16_RNE_SCALAR_F0_TO(reg) \
    FMOVS F0, reg; \
    MOVD $0x7fff, R8; \
    LSR $16, reg, R9; \
    AND $1, R9, R9; \
    ADD R8, R9, R9; \
    ADD R9, reg, reg; \
    LSR $16, reg, reg
