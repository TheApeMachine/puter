// SPDX-License-Identifier: Apache-2.0
// NEON active inference bfloat16 kernels: VSHLL widen, BFCVTN narrow, f32 math.
#include "textflag.h"

DATA aiLogC<>+0(SB)/4, $0.6931471805599453
DATA aiLogC<>+4(SB)/4, $1.0
DATA aiLogC<>+8(SB)/4, $0.09090909
DATA aiLogC<>+12(SB)/4, $0.11111111
DATA aiLogC<>+16(SB)/4, $0.14285715
DATA aiLogC<>+20(SB)/4, $0.20000000
DATA aiLogC<>+24(SB)/4, $0.33333334
DATA aiLogC<>+28(SB)/4, $2.0
GLOBL aiLogC<>(SB), 8, $32

#define VFADD_S4(m, n, d)   WORD $(0x4E20D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFSUB_S4(m, n, d)   WORD $(0x4EA0D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_S4(m, n, d)   WORD $(0x6E20DC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFMLA_S4(m, n, d)   WORD $(0x4E20CC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFDIV_S4(m, n, d)   WORD $(0x6E20FC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFMLA_D2(m, n, d)   WORD $(0x4E60CC00 | ((m) << 16) | ((n) << 5) | (d))
#define FCVTL_2D(n, d)      WORD $(0x0E617800 | ((n) << 5) | (d))
#define FCVTL2_2D(n, d)     WORD $(0x4E617800 | ((n) << 5) | (d))
#define VFADD_D2(m, n, d)   WORD $(0x4E60D400 | ((m) << 16) | ((n) << 5) | (d))
#define FADDP_D(n, d)       WORD $(0x7E70D800 | ((n) << 5) | (d))
#define VUSHR_S4_BY23(n, d) WORD $(0x6F290400 | ((n) << 5) | (d))
#define VISUB_S4(m, n, d)   WORD $(0x6EA08400 | ((m) << 16) | ((n) << 5) | (d))
#define VAND_B16(m, n, d)   WORD $(0x4E201C00 | ((m) << 16) | ((n) << 5) | (d))
#define VORR_B16(m, n, d)   WORD $(0x4EA01C00 | ((m) << 16) | ((n) << 5) | (d))
#define VSCVTF_S4(n, d)     WORD $(0x4E21D800 | ((n) << 5) | (d))
#define VMAX_S4(m, n, d)    WORD $(0x4E20F400 | ((m) << 16) | ((n) << 5) | (d))
#define VMOV_B16(src, dst)  WORD $(0x4EA01C00 | ((src) << 16) | ((src) << 5) | (dst))

#define VSHLL_S4_H4_16(n, d)  WORD $(0x2E613800 | ((n) << 5) | (d))
#define VSHLL2_S4_H8_16(n, d) WORD $(0x6E613800 | ((n) << 5) | (d))
#define BFCVTN_H4_S4(n, d)    WORD $(0x0EA16800 | ((n) << 5) | (d))
#define BFCVTN2_H8_S4(n, d)   WORD $(0x4EA16800 | ((n) << 5) | (d))

#define BF16_WIDEN_H8_TO_S4_LOW(src, dst)  VZIP1 src, V30.H8, dst
#define BF16_WIDEN_H8_TO_S4_HIGH(src, dst) VZIP2 src, V30.H8, dst
#define BF16_NARROW_S4_TO_H8(s0, s1, dst) VUZP2 s1, s0, dst

#define AI_F32X4_TO_F64_ADD_F14(src) \
	FCVTL_2D((src), 12) ;\
	FCVTL2_2D((src), 13) ;\
	FADDP_D(12, 12) ;\
	FADDP_D(13, 13) ;\
	FADDD F12, F14, F14 ;\
	FADDD F13, F14, F14

#define AI_F32X4_TO_F64_ADD_CE(src) \
	FCVTL_2D(src, 8) ;\
	FCVTL2_2D(src, 9) ;\
	VFADD_D2(8, 28, 28) ;\
	VFADD_D2(9, 28, 28)

#define AI_F32X4_TO_F64_ADD_KL(src) \
	FCVTL_2D(src, 8) ;\
	FCVTL2_2D(src, 9) ;\
	VFADD_D2(8, 29, 29) ;\
	VFADD_D2(9, 29, 29)

#define AI_F32_LANE0_F64_ADD_CE \
	FCVTSD F13, F6 ;\
	FADDD F6, F14, F14

#define AI_F32_LANE0_V13_F64_ADD_KL \
	FCVTSD F13, F6 ;\
	FADDD F6, F15, F15

#define AI_F32_LANE0_V14_F64_ADD_KL \
	FCVTSD F14, F6 ;\
	FADDD F6, F15, F15

#define AI_LOAD_LOG_MASKS \
	MOVD $0x007FFFFF, R6 ;\
	VMOV R6, V24.S[0] ;\
	VDUP V24.S[0], V24.S4 ;\
	MOVD $0x3F800000, R6 ;\
	VMOV R6, V25.S[0] ;\
	VDUP V25.S[0], V25.S4 ;\
	MOVD $127, R6 ;\
	VMOV R6, V26.S[0] ;\
	VDUP V26.S[0], V26.S4

#define AI_RELOAD_LOG_POLY \
	MOVD $aiLogC<>(SB), R14 ;\
	FMOVS  0(R14), F16 ;\
	VDUP V16.S[0], V16.S4 ;\
	FMOVS  4(R14), F17 ;\
	VDUP V17.S[0], V17.S4 ;\
	FMOVS  8(R14), F18 ;\
	VDUP V18.S[0], V18.S4 ;\
	FMOVS 12(R14), F19 ;\
	VDUP V19.S[0], V19.S4 ;\
	FMOVS 16(R14), F20 ;\
	VDUP V20.S[0], V20.S4 ;\
	FMOVS 20(R14), F21 ;\
	VDUP V21.S[0], V21.S4 ;\
	FMOVS 24(R14), F22 ;\
	VDUP V22.S[0], V22.S4 ;\
	FMOVS 28(R14), F23 ;\
	VDUP V23.S[0], V23.S4

#define AI_LOAD_LOG_CONSTS \
	AI_RELOAD_LOG_POLY ;\
	AI_LOAD_LOG_MASKS

#define AI_NEON_LOG4(in, out) \
	VUSHR_S4_BY23(in, 1) ;\
	VISUB_S4(26, 1, 1) ;\
	VAND_B16(24, in, 2) ;\
	VORR_B16(25, 2, 2) ;\
	VSCVTF_S4(1, 1) ;\
	VFSUB_S4(17, 2, 3) ;\
	VFADD_S4(17, 2, 4) ;\
	VFDIV_S4(4, 3, 5) ;\
	VFMUL_S4(5, 5, 6) ;\
	VMOV_B16(18, 7) ;\
	VMOV_B16(19, 8) ;\
	VFMLA_S4(6, 7, 8) ;\
	VMOV_B16(20, 7) ;\
	VFMLA_S4(6, 8, 7) ;\
	VMOV_B16(21, 8) ;\
	VFMLA_S4(6, 7, 8) ;\
	VMOV_B16(22, 7) ;\
	VFMLA_S4(6, 8, 7) ;\
	VMOV_B16(17, 8) ;\
	VFMLA_S4(6, 7, 8) ;\
	VFMUL_S4(5, 8, 8) ;\
	VFMUL_S4(23, 8, 8) ;\
	VFMLA_S4(16, 1, 8) ;\
	VMOV_B16(8, out)

#define AI_BF16_STORE_F32_AS_BF16 \
	FMOVS F0, R6 ;\
	LSR  $16, R6, R6 ;\
	MOVH R6, (R2)

#define AI_BF16_FE_PROCESS4 \
	VMOV_B16(4, 27) ;\
	VMAX_S4(31, 3, 3) ;\
	VMAX_S4(31, 4, 4) ;\
	VMAX_S4(31, 5, 5) ;\
	AI_RELOAD_LOG_POLY ;\
	AI_NEON_LOG4(3, 10) ;\
	AI_RELOAD_LOG_POLY ;\
	AI_NEON_LOG4(4, 11) ;\
	AI_RELOAD_LOG_POLY ;\
	AI_NEON_LOG4(5, 12) ;\
	VEOR V30.B16, V30.B16, V30.B16 ;\
	VFSUB_S4(10, 30, 13) ;\
	VFMUL_S4(27, 13, 13) ;\
	AI_F32X4_TO_F64_ADD_CE(13) ;\
	VFSUB_S4(12, 11, 14) ;\
	VFMUL_S4(27, 14, 14) ;\
	AI_F32X4_TO_F64_ADD_KL(14)

#define AI_BF16_FE_HALF \
	BF16_WIDEN_H8_TO_S4_LOW(V3.H8, V6.H8) ;\
	BF16_WIDEN_H8_TO_S4_LOW(V4.H8, V7.H8) ;\
	BF16_WIDEN_H8_TO_S4_LOW(V5.H8, V8.H8) ;\
	VMOV_B16(6, 3) ;\
	VMOV_B16(7, 4) ;\
	VMOV_B16(8, 5) ;\
	AI_BF16_FE_PROCESS4 ;\
	BF16_WIDEN_H8_TO_S4_HIGH(V3.H8, V6.H8) ;\
	BF16_WIDEN_H8_TO_S4_HIGH(V4.H8, V7.H8) ;\
	BF16_WIDEN_H8_TO_S4_HIGH(V5.H8, V8.H8) ;\
	VMOV_B16(6, 3) ;\
	VMOV_B16(7, 4) ;\
	VMOV_B16(8, 5) ;\
	AI_BF16_FE_PROCESS4

#define AI_BF16_FE_BLOCK8 \
	VLD1 (R10), [V3.H8] ;\
	VLD1 (R11), [V4.H8] ;\
	VLD1 (R12), [V5.H8] ;\
	AI_BF16_FE_HALF ;\
	ADD  $16, R10 ;\
	ADD  $16, R11 ;\
	ADD  $16, R12 ;\
	SUB  $8, R3

#define AI_BF16_FE_TAIL_ONE \
	MOVHU (R10), R6 ;\
	LSL  $16, R6, R6 ;\
	FMOVS R6, F3 ;\
	VDUP V3.S[0], V3.S4 ;\
	MOVHU (R11), R6 ;\
	LSL  $16, R6, R6 ;\
	FMOVS R6, F4 ;\
	VDUP V4.S[0], V4.S4 ;\
	MOVHU (R12), R6 ;\
	LSL  $16, R6, R6 ;\
	FMOVS R6, F5 ;\
	VDUP V5.S[0], V5.S4 ;\
	VMOV_B16(4, 27) ;\
	VMAX_S4(31, 3, 3) ;\
	VMAX_S4(31, 4, 4) ;\
	VMAX_S4(31, 5, 5) ;\
	AI_RELOAD_LOG_POLY ;\
	AI_NEON_LOG4(3, 10) ;\
	AI_RELOAD_LOG_POLY ;\
	AI_NEON_LOG4(4, 11) ;\
	AI_RELOAD_LOG_POLY ;\
	AI_NEON_LOG4(5, 12) ;\
	VEOR V30.B16, V30.B16, V30.B16 ;\
	VFSUB_S4(10, 30, 13) ;\
	VFMUL_S4(27, 13, 13) ;\
	AI_F32_LANE0_F64_ADD_CE ;\
	VFSUB_S4(12, 11, 14) ;\
	VFMUL_S4(27, 14, 14) ;\
	AI_F32_LANE0_V14_F64_ADD_KL ;\
	ADD  $2, R10 ;\
	ADD  $2, R11 ;\
	ADD  $2, R12 ;\
	SUB  $1, R3

#define AI_BF16_EFE_OBS_BLOCK8 \
	VLD1 (R0), [V3.H8] ;\
	VLD1 (R1), [V4.H8] ;\
	BF16_WIDEN_H8_TO_S4_LOW(V3.H8, V6.H8) ;\
	BF16_WIDEN_H8_TO_S4_LOW(V4.H8, V7.H8) ;\
	VMOV_B16(6, 3) ;\
	VMOV_B16(7, 4) ;\
	VMOV_B16(3, 27) ;\
	VMAX_S4(31, 3, 3) ;\
	VMAX_S4(31, 4, 4) ;\
	AI_RELOAD_LOG_POLY ;\
	AI_NEON_LOG4(3, 10) ;\
	AI_RELOAD_LOG_POLY ;\
	AI_NEON_LOG4(4, 11) ;\
	VFSUB_S4(11, 10, 13) ;\
	VFMUL_S4(27, 13, 13) ;\
	AI_F32X4_TO_F64_ADD_CE(13) ;\
	BF16_WIDEN_H8_TO_S4_HIGH(V3.H8, V6.H8) ;\
	BF16_WIDEN_H8_TO_S4_HIGH(V4.H8, V7.H8) ;\
	VMOV_B16(6, 3) ;\
	VMOV_B16(7, 4) ;\
	VMOV_B16(3, 27) ;\
	VMAX_S4(31, 3, 3) ;\
	VMAX_S4(31, 4, 4) ;\
	AI_RELOAD_LOG_POLY ;\
	AI_NEON_LOG4(3, 10) ;\
	AI_RELOAD_LOG_POLY ;\
	AI_NEON_LOG4(4, 11) ;\
	VFSUB_S4(11, 10, 13) ;\
	VFMUL_S4(27, 13, 13) ;\
	AI_F32X4_TO_F64_ADD_CE(13) ;\
	ADD  $16, R0 ;\
	ADD  $16, R1 ;\
	SUB  $8, R3

#define AI_BF16_EFE_OBS_TAIL_ONE \
	MOVHU (R0), R6 ;\
	LSL  $16, R6, R6 ;\
	FMOVS R6, F3 ;\
	VDUP V3.S[0], V3.S4 ;\
	MOVHU (R1), R6 ;\
	LSL  $16, R6, R6 ;\
	FMOVS R6, F4 ;\
	VDUP V4.S[0], V4.S4 ;\
	VMOV_B16(3, 27) ;\
	VMAX_S4(31, 3, 3) ;\
	VMAX_S4(31, 4, 4) ;\
	AI_RELOAD_LOG_POLY ;\
	AI_NEON_LOG4(3, 10) ;\
	AI_RELOAD_LOG_POLY ;\
	AI_NEON_LOG4(4, 11) ;\
	VFSUB_S4(11, 10, 13) ;\
	VFMUL_S4(27, 13, 13) ;\
	AI_F32_LANE0_F64_ADD_CE ;\
	ADD  $2, R0 ;\
	ADD  $2, R1 ;\
	SUB  $1, R3

#define AI_BF16_EFE_STATE_BLOCK8 \
	VLD1 (R2), [V3.H8] ;\
	BF16_WIDEN_H8_TO_S4_LOW(V3.H8, V6.H8) ;\
	VMOV_B16(6, 3) ;\
	VMOV_B16(3, 27) ;\
	VMAX_S4(31, 3, 3) ;\
	AI_RELOAD_LOG_POLY ;\
	AI_NEON_LOG4(3, 10) ;\
	VEOR V30.B16, V30.B16, V30.B16 ;\
	VFSUB_S4(10, 30, 13) ;\
	VFMUL_S4(27, 13, 13) ;\
	AI_F32X4_TO_F64_ADD_KL(13) ;\
	BF16_WIDEN_H8_TO_S4_HIGH(V3.H8, V6.H8) ;\
	VMOV_B16(6, 3) ;\
	VMOV_B16(3, 27) ;\
	VMAX_S4(31, 3, 3) ;\
	AI_RELOAD_LOG_POLY ;\
	AI_NEON_LOG4(3, 10) ;\
	VEOR V30.B16, V30.B16, V30.B16 ;\
	VFSUB_S4(10, 30, 13) ;\
	VFMUL_S4(27, 13, 13) ;\
	AI_F32X4_TO_F64_ADD_KL(13) ;\
	ADD  $16, R2 ;\
	SUB  $8, R4

#define AI_BF16_EFE_STATE_TAIL_ONE \
	MOVHU (R2), R6 ;\
	LSL  $16, R6, R6 ;\
	FMOVS R6, F3 ;\
	VDUP V3.S[0], V3.S4 ;\
	VMOV_B16(3, 27) ;\
	VMAX_S4(31, 3, 3) ;\
	AI_RELOAD_LOG_POLY ;\
	AI_NEON_LOG4(3, 10) ;\
	VEOR V30.B16, V30.B16, V30.B16 ;\
	VFSUB_S4(10, 30, 13) ;\
	VFMUL_S4(27, 13, 13) ;\
	AI_F32_LANE0_V13_F64_ADD_KL ;\
	ADD  $2, R2 ;\
	SUB  $1, R4

#define AI_BF16_STORE_RESULT \
	FADDP_D(28, 0) ;\
	FADDP_D(29, 1) ;\
	FADDD F14, F0, F0 ;\
	FADDD F15, F1, F1 ;\
	FADDD F1, F0, F0 ;\
	FCVTDS F0, F0 ;\
	FMOVS F0, R6 ;\
	LSR  $16, R6, R6 ;\
	MOVH R6, ret+32(FP)

// func PrecisionWeightBFloat16NEONAsm(errors, precision, output *uint16, count int)
TEXT ·PrecisionWeightBFloat16NEONAsm(SB), NOSPLIT, $0-32
	MOVD errors+0(FP), R0
	MOVD precision+8(FP), R1
	MOVD output+16(FP), R2
	MOVD count+24(FP), R3
	CBZ  R3, ai_bf16_pw_done

	VEOR V30.B16, V30.B16, V30.B16

ai_bf16_pw_loop16:
	CMP  $16, R3
	BLT  ai_bf16_pw_loop8

	VLD1.P 32(R0), [V0.H8, V1.H8]
	VLD1.P 32(R1), [V2.H8, V3.H8]
	BF16_WIDEN_H8_TO_S4_LOW(V0.H8, V4.H8)
	BF16_WIDEN_H8_TO_S4_HIGH(V0.H8, V5.H8)
	BF16_WIDEN_H8_TO_S4_LOW(V1.H8, V6.H8)
	BF16_WIDEN_H8_TO_S4_HIGH(V1.H8, V7.H8)
	BF16_WIDEN_H8_TO_S4_LOW(V2.H8, V8.H8)
	BF16_WIDEN_H8_TO_S4_HIGH(V2.H8, V9.H8)
	BF16_WIDEN_H8_TO_S4_LOW(V3.H8, V10.H8)
	BF16_WIDEN_H8_TO_S4_HIGH(V3.H8, V11.H8)
	VFMUL_S4(8, 4, 4)
	VFMUL_S4(9, 5, 5)
	VFMUL_S4(10, 6, 6)
	VFMUL_S4(11, 7, 7)
	BF16_NARROW_S4_TO_H8(V4.H8, V5.H8, V12.H8)
	BF16_NARROW_S4_TO_H8(V6.H8, V7.H8, V13.H8)
	VST1.P [V12.H8, V13.H8], 32(R2)
	SUB  $16, R3
	B    ai_bf16_pw_loop16

ai_bf16_pw_loop8:
	CMP  $8, R3
	BLT  ai_bf16_pw_scalar

	VLD1.P 16(R0), [V0.H8]
	VLD1.P 16(R1), [V2.H8]
	BF16_WIDEN_H8_TO_S4_LOW(V0.H8, V4.H8)
	BF16_WIDEN_H8_TO_S4_HIGH(V0.H8, V5.H8)
	BF16_WIDEN_H8_TO_S4_LOW(V2.H8, V8.H8)
	BF16_WIDEN_H8_TO_S4_HIGH(V2.H8, V9.H8)
	VFMUL_S4(8, 4, 4)
	VFMUL_S4(9, 5, 5)
	BF16_NARROW_S4_TO_H8(V4.H8, V5.H8, V12.H8)
	VST1.P [V12.H8], 16(R2)
	SUB  $8, R3
	B    ai_bf16_pw_loop8

ai_bf16_pw_scalar:
	CBZ  R3, ai_bf16_pw_done

ai_bf16_pw_scalar_loop:
	MOVHU (R0), R4
	MOVHU (R1), R5
	LSL  $16, R4, R4
	LSL  $16, R5, R5
	FMOVS R4, F0
	FMOVS R5, F1
	FMULS F1, F0, F0
	AI_BF16_STORE_F32_AS_BF16
	ADD  $2, R0
	ADD  $2, R1
	ADD  $2, R2
	SUB  $1, R3
	CBNZ R3, ai_bf16_pw_scalar_loop

ai_bf16_pw_done:
	RET

#define AI_BF16_BU_ACCUM_V28(src) \
	FCVTL_2D((src), 12) ;\
	FCVTL2_2D((src), 13) ;\
	VFADD_D2(12, 28, 28) ;\
	VFADD_D2(13, 28, 28)

// func BeliefUpdateBFloat16NEONAsm(likelihood, prior, output *uint16, count int)
TEXT ·BeliefUpdateBFloat16NEONAsm(SB), NOSPLIT, $8-32
	MOVD likelihood+0(FP), R0
	MOVD prior+8(FP), R1
	MOVD output+16(FP), R2
	MOVD count+24(FP), R3
	CBZ  R3, ai_bf16_bu_done

	VEOR V30.B16, V30.B16, V30.B16

ai_bf16_bu_store_loop16:
	CMP  $16, R3
	BLT  ai_bf16_bu_store_loop8

	VLD1.P 32(R0), [V0.H8, V1.H8]
	VLD1.P 32(R1), [V2.H8, V3.H8]
	BF16_WIDEN_H8_TO_S4_LOW(V0.H8, V4.H8)
	BF16_WIDEN_H8_TO_S4_HIGH(V0.H8, V5.H8)
	BF16_WIDEN_H8_TO_S4_LOW(V1.H8, V6.H8)
	BF16_WIDEN_H8_TO_S4_HIGH(V1.H8, V7.H8)
	BF16_WIDEN_H8_TO_S4_LOW(V2.H8, V8.H8)
	BF16_WIDEN_H8_TO_S4_HIGH(V2.H8, V9.H8)
	BF16_WIDEN_H8_TO_S4_LOW(V3.H8, V10.H8)
	BF16_WIDEN_H8_TO_S4_HIGH(V3.H8, V11.H8)
	VFMUL_S4(8, 4, 4)
	VFMUL_S4(9, 5, 5)
	VFMUL_S4(10, 6, 6)
	VFMUL_S4(11, 7, 7)
	BF16_NARROW_S4_TO_H8(V4.H8, V5.H8, V12.H8)
	BF16_NARROW_S4_TO_H8(V6.H8, V7.H8, V13.H8)
	VST1.P [V12.H8, V13.H8], 32(R2)
	SUB  $16, R3
	B    ai_bf16_bu_store_loop16

ai_bf16_bu_store_loop8:
	CMP  $8, R3
	BLT  ai_bf16_bu_store_scalar

	VLD1.P 16(R0), [V0.H8]
	VLD1.P 16(R1), [V2.H8]
	BF16_WIDEN_H8_TO_S4_LOW(V0.H8, V4.H8)
	BF16_WIDEN_H8_TO_S4_HIGH(V0.H8, V5.H8)
	BF16_WIDEN_H8_TO_S4_LOW(V2.H8, V8.H8)
	BF16_WIDEN_H8_TO_S4_HIGH(V2.H8, V9.H8)
	VFMUL_S4(8, 4, 4)
	VFMUL_S4(9, 5, 5)
	BF16_NARROW_S4_TO_H8(V4.H8, V5.H8, V12.H8)
	VST1.P [V12.H8], 16(R2)
	SUB  $8, R3
	B    ai_bf16_bu_store_loop8

ai_bf16_bu_store_scalar:
	CBZ  R3, ai_bf16_bu_sum

ai_bf16_bu_store_scalar_loop:
	MOVHU (R0), R4
	MOVHU (R1), R5
	LSL  $16, R4, R4
	LSL  $16, R5, R5
	FMOVS R4, F0
	FMOVS R5, F1
	FMULS F1, F0, F0
	AI_BF16_STORE_F32_AS_BF16
	ADD  $2, R0
	ADD  $2, R1
	ADD  $2, R2
	SUB  $1, R3
	CBNZ R3, ai_bf16_bu_store_scalar_loop

ai_bf16_bu_sum:
	MOVD likelihood+0(FP), R0
	MOVD prior+8(FP), R1
	MOVD count+24(FP), R3
	FMOVD $0, F25

ai_bf16_bu_sum_loop:
	CBZ  R3, ai_bf16_bu_normalize

	MOVHU (R0), R4
	MOVHU (R1), R5
	LSL  $16, R4, R4
	LSL  $16, R5, R5
	FMOVS R4, F0
	FMOVS R5, F1
	FMULS F1, F0, F0
	FCVTSD F0, F6
	FADDD F6, F25, F25
	ADD  $2, R0
	ADD  $2, R1
	SUB  $1, R3
	B    ai_bf16_bu_sum_loop

ai_bf16_bu_normalize:
	FMOVD $0, F15
	FCMPD F25, F15
	BEQ  ai_bf16_bu_done

	FMOVD $1.0, F3
	FDIVD F25, F3, F3
	FCVTDS F3, F3
	FMOVS F3, ai_bf16_bu_norm-8(SP)
	FMOVS ai_bf16_bu_norm-8(SP), F28
	VDUP V28.S[0], V28.S4

	MOVD output+16(FP), R2
	MOVD count+24(FP), R3
	VEOR V30.B16, V30.B16, V30.B16

ai_bf16_bu_scale_loop16:
	CMP  $16, R3
	BLT  ai_bf16_bu_scale_loop8

	VLD1.P 32(R2), [V0.H8, V1.H8]
	BF16_WIDEN_H8_TO_S4_LOW(V0.H8, V4.H8)
	BF16_WIDEN_H8_TO_S4_HIGH(V0.H8, V5.H8)
	BF16_WIDEN_H8_TO_S4_LOW(V1.H8, V6.H8)
	BF16_WIDEN_H8_TO_S4_HIGH(V1.H8, V7.H8)
	VFMUL_S4(28, 4, 4)
	VFMUL_S4(28, 5, 5)
	VFMUL_S4(28, 6, 6)
	VFMUL_S4(28, 7, 7)
	BF16_NARROW_S4_TO_H8(V4.H8, V5.H8, V12.H8)
	BF16_NARROW_S4_TO_H8(V6.H8, V7.H8, V13.H8)
	VST1.P [V12.H8, V13.H8], 32(R2)
	SUB  $16, R3
	B    ai_bf16_bu_scale_loop16

ai_bf16_bu_scale_loop8:
	CMP  $8, R3
	BLT  ai_bf16_bu_scale_scalar

	VLD1.P 16(R2), [V0.H8]
	BF16_WIDEN_H8_TO_S4_LOW(V0.H8, V4.H8)
	BF16_WIDEN_H8_TO_S4_HIGH(V0.H8, V5.H8)
	VFMUL_S4(28, 4, 4)
	VFMUL_S4(28, 5, 5)
	BF16_NARROW_S4_TO_H8(V4.H8, V5.H8, V12.H8)
	VST1.P [V12.H8], 16(R2)
	SUB  $8, R3
	B    ai_bf16_bu_scale_loop8

ai_bf16_bu_scale_scalar:
	CBZ  R3, ai_bf16_bu_done

ai_bf16_bu_scale_scalar_loop:
	MOVHU (R2), R4
	LSL  $16, R4, R4
	FMOVS R4, F0
	FMULS F3, F0, F0
	AI_BF16_STORE_F32_AS_BF16
	ADD  $2, R2
	SUB  $1, R3
	CBNZ R3, ai_bf16_bu_scale_scalar_loop

ai_bf16_bu_done:
	RET

// func FreeEnergyBFloat16NEONAsm(likelihood, posterior, prior *uint16, count int) uint16
TEXT ·FreeEnergyBFloat16NEONAsm(SB), NOSPLIT, $0-34
	MOVD likelihood+0(FP), R10
	MOVD posterior+8(FP), R11
	MOVD prior+16(FP), R12
	MOVD count+24(FP), R3

	MOVD $0x2b8cbccc, R17
	VMOV R17, V31.S[0]
	VDUP V31.S[0], V31.S4
	VEOR V30.B16, V30.B16, V30.B16
	AI_LOAD_LOG_CONSTS
	VEOR V28.B16, V28.B16, V28.B16
	VEOR V29.B16, V29.B16, V29.B16
	FMOVD $0, F14
	FMOVD $0, F15
	CBZ  R3, ai_bf16_fe_store

ai_bf16_fe_loop8:
	CMP  $8, R3
	BLT  ai_bf16_fe_tail

	AI_BF16_FE_BLOCK8
	B    ai_bf16_fe_loop8

ai_bf16_fe_tail:
	CBZ  R3, ai_bf16_fe_store

ai_bf16_fe_tail_loop:
	AI_BF16_FE_TAIL_ONE
	CBNZ R3, ai_bf16_fe_tail_loop

ai_bf16_fe_store:
	AI_BF16_STORE_RESULT
	RET

// func ExpectedFreeEnergyBFloat16NEONAsm(
//     predictedObs, preferredObs, predictedState *uint16,
//     obsCount, stateCount int,
// ) uint16
TEXT ·ExpectedFreeEnergyBFloat16NEONAsm(SB), NOSPLIT, $0-42
	MOVD predictedObs+0(FP), R0
	MOVD preferredObs+8(FP), R1
	MOVD predictedState+16(FP), R2
	MOVD obsCount+24(FP), R3
	MOVD stateCount+32(FP), R4

	MOVD $0x2b8cbccc, R17
	VMOV R17, V31.S[0]
	VDUP V31.S[0], V31.S4
	VEOR V30.B16, V30.B16, V30.B16
	AI_LOAD_LOG_CONSTS
	VEOR V28.B16, V28.B16, V28.B16
	VEOR V29.B16, V29.B16, V29.B16
	FMOVD $0, F14
	FMOVD $0, F15

ai_bf16_efe_obs_loop8:
	CMP  $8, R3
	BLT  ai_bf16_efe_obs_tail

	AI_BF16_EFE_OBS_BLOCK8
	B    ai_bf16_efe_obs_loop8

ai_bf16_efe_obs_tail:
	CBZ  R3, ai_bf16_efe_state_init

ai_bf16_efe_obs_tail_loop:
	AI_BF16_EFE_OBS_TAIL_ONE
	CBNZ R3, ai_bf16_efe_obs_tail_loop

ai_bf16_efe_state_init:
	MOVD predictedState+16(FP), R2
	MOVD stateCount+32(FP), R4

ai_bf16_efe_state_loop8:
	CMP  $8, R4
	BLT  ai_bf16_efe_state_tail

	AI_BF16_EFE_STATE_BLOCK8
	B    ai_bf16_efe_state_loop8

ai_bf16_efe_state_tail:
	CBZ  R4, ai_bf16_efe_store

ai_bf16_efe_state_tail_loop:
	AI_BF16_EFE_STATE_TAIL_ONE
	CBNZ R4, ai_bf16_efe_state_tail_loop

ai_bf16_efe_store:
	FADDP_D(28, 0)
	FADDP_D(29, 1)
	FADDD F14, F0, F0
	FADDD F15, F1, F1
	FADDD F1, F0, F0
	FCVTDS F0, F0
	FMOVS F0, R6
	LSR  $16, R6, R6
	MOVH R6, ret+40(FP)
	RET
