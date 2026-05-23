// SPDX-License-Identifier: Apache-2.0
// NEON active inference float16 kernels: widen fp16→f32, compute, narrow.
#include "textflag.h"

DATA aiLogC<>+0(SB)/4, $0.6931471805599453
DATA aiLogC<>+4(SB)/4, $1.0
DATA aiLogC<>+8(SB)/4, $0.09090909
DATA aiLogC<>+12(SB)/4, $0.11111111
DATA aiLogC<>+16(SB)/4, $0.14285715
DATA aiLogC<>+20(SB)/4, $0.20000000
DATA aiLogC<>+24(SB)/4, $0.33333334
DATA aiLogC<>+28(SB)/4, $2.0
GLOBL aiLogC<>(SB), RODATA|NOPTR, $32

#define VFCVTL_4S(n, d)   WORD $(0x0E217800 | ((n) << 5) | (d))
#define VFCVTL2_4S(n, d)  WORD $(0x4E217800 | ((n) << 5) | (d))
#define VFCVTN_4H(n, d)   WORD $(0x0E216800 | ((n) << 5) | (d))
#define VFCVTN2_8H(n, d)  WORD $(0x4E216800 | ((n) << 5) | (d))
#define VFADD_S4(m, n, d) WORD $(0x4E20D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFSUB_S4(m, n, d) WORD $(0x4EA0D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_S4(m, n, d) WORD $(0x6E20DC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFMLA_S4(m, n, d) WORD $(0x4E20CC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFDIV_S4(m, n, d) WORD $(0x6E20FC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFMLA_D2(m, n, d) WORD $(0x4E60CC00 | ((m) << 16) | ((n) << 5) | (d))
#define FCVTL_2D(n, d)    WORD $(0x0E617800 | ((n) << 5) | (d))
#define FCVTL2_2D(n, d)   WORD $(0x4E617800 | ((n) << 5) | (d))
#define VFADD_D2(m, n, d) WORD $(0x4E60D400 | ((m) << 16) | ((n) << 5) | (d))
#define FADDP_D(n, d)     WORD $(0x7E70D800 | ((n) << 5) | (d))
#define VUSHR_S4_BY23(n, d) WORD $(0x6F290400 | ((n) << 5) | (d))
#define VISUB_S4(m, n, d) WORD $(0x6EA08400 | ((m) << 16) | ((n) << 5) | (d))
#define VAND_B16(m, n, d) WORD $(0x4E201C00 | ((m) << 16) | ((n) << 5) | (d))
#define VORR_B16(m, n, d) WORD $(0x4EA01C00 | ((m) << 16) | ((n) << 5) | (d))
#define VSCVTF_S4(n, d)   WORD $(0x4E21D800 | ((n) << 5) | (d))
#define VMAX_S4(m, n, d)  WORD $(0x4E20F400 | ((m) << 16) | ((n) << 5) | (d))
#define VMOV_B16(src, dst) WORD $(0x4EA01C00 | ((src) << 16) | ((src) << 5) | (dst))

#define AI_F32X4_TO_F64_ADD_F14(src) \
	FCVTL_2D(src, 8) ;\
	FCVTL2_2D(src, 9) ;\
	FADDD F8, F14, F14 ;\
	FADDD F9, F14, F14 ;\
	FADDD F10, F14, F14 ;\
	FADDD F11, F14, F14

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

#define AI_FP16_WIDEN_LOW(hreg, sreg)  VFCVTL_4S(hreg, sreg)
#define AI_FP16_WIDEN_HIGH(hreg, sreg) VFCVTL2_4S(hreg, sreg)

#define AI_FP16_STORE_F32_AS_F16 \
	FCVTSH F0, F0 ;\
	FMOVD F0, R6 ;\
	MOVH R6, (R2)

#define AI_FP16_FE_PROCESS4 \
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

#define AI_FP16_FE_HALF(hlike, hpost, hprior) \
	AI_FP16_WIDEN_LOW(hlike, 3) ;\
	AI_FP16_WIDEN_LOW(hpost, 4) ;\
	AI_FP16_WIDEN_LOW(hprior, 5) ;\
	AI_FP16_FE_PROCESS4 ;\
	AI_FP16_WIDEN_HIGH(hlike, 3) ;\
	AI_FP16_WIDEN_HIGH(hpost, 4) ;\
	AI_FP16_WIDEN_HIGH(hprior, 5) ;\
	AI_FP16_FE_PROCESS4

#define AI_FP16_FE_BLOCK8 \
	VLD1 (R10), [V3.H8] ;\
	VLD1 (R11), [V4.H8] ;\
	VLD1 (R12), [V5.H8] ;\
	AI_FP16_FE_HALF(3, 4, 5) ;\
	ADD  $16, R10 ;\
	ADD  $16, R11 ;\
	ADD  $16, R12 ;\
	SUB  $8, R3

#define AI_FP16_FE_TAIL_ONE \
	MOVHU (R10), R6 ;\
	VMOV R6, V3.H[0] ;\
	VFCVTL_4S(3, 3) ;\
	VDUP V3.S[0], V3.S4 ;\
	MOVHU (R11), R6 ;\
	VMOV R6, V4.H[0] ;\
	VFCVTL_4S(4, 4) ;\
	VDUP V4.S[0], V4.S4 ;\
	MOVHU (R12), R6 ;\
	VMOV R6, V5.H[0] ;\
	VFCVTL_4S(5, 5) ;\
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

#define AI_FP16_EFE_OBS_BLOCK8 \
	VLD1 (R0), [V3.H8] ;\
	VLD1 (R1), [V4.H8] ;\
	AI_FP16_WIDEN_LOW(3, 3) ;\
	AI_FP16_WIDEN_LOW(4, 4) ;\
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
	AI_FP16_WIDEN_HIGH(3, 3) ;\
	AI_FP16_WIDEN_HIGH(4, 4) ;\
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

#define AI_FP16_EFE_OBS_TAIL_ONE \
	MOVHU (R0), R6 ;\
	VMOV R6, V3.H[0] ;\
	VFCVTL_4S(3, 3) ;\
	VDUP V3.S[0], V3.S4 ;\
	MOVHU (R1), R6 ;\
	VMOV R6, V4.H[0] ;\
	VFCVTL_4S(4, 4) ;\
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

#define AI_FP16_EFE_STATE_BLOCK8 \
	VLD1 (R2), [V3.H8] ;\
	AI_FP16_WIDEN_LOW(3, 3) ;\
	VMOV_B16(3, 27) ;\
	VMAX_S4(31, 3, 3) ;\
	AI_RELOAD_LOG_POLY ;\
	AI_NEON_LOG4(3, 10) ;\
	VEOR V30.B16, V30.B16, V30.B16 ;\
	VFSUB_S4(10, 30, 13) ;\
	VFMUL_S4(27, 13, 13) ;\
	AI_F32X4_TO_F64_ADD_KL(13) ;\
	AI_FP16_WIDEN_HIGH(3, 3) ;\
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

#define AI_FP16_EFE_STATE_TAIL_ONE \
	MOVHU (R2), R6 ;\
	VMOV R6, V3.H[0] ;\
	VFCVTL_4S(3, 3) ;\
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

#define AI_FP16_STORE_RESULT \
	FADDP_D(28, 0) ;\
	FADDP_D(29, 1) ;\
	FADDD F14, F0, F0 ;\
	FADDD F15, F1, F1 ;\
	FADDD F1, F0, F0 ;\
	FCVTDS F0, F0 ;\
	FCVTSH F0, F0 ;\
	FMOVD F0, R6 ;\
	MOVH R6, ret+32(FP)

// func PrecisionWeightFloat16NEONAsm(errors, precision, output *uint16, count int)
TEXT ·PrecisionWeightFloat16NEONAsm(SB), NOSPLIT, $0-32
	MOVD errors+0(FP), R0
	MOVD precision+8(FP), R1
	MOVD output+16(FP), R2
	MOVD count+24(FP), R3
	CBZ  R3, ai_fp16_pw_done

ai_fp16_pw_loop16:
	CMP  $16, R3
	BLT  ai_fp16_pw_loop8

	VLD1.P 32(R0), [V0.H8, V1.H8]
	VLD1.P 32(R1), [V2.H8, V3.H8]
	VFCVTL_4S(0, 4)
	VFCVTL2_4S(0, 5)
	VFCVTL_4S(1, 6)
	VFCVTL2_4S(1, 7)
	VFCVTL_4S(2, 8)
	VFCVTL2_4S(2, 9)
	VFCVTL_4S(3, 10)
	VFCVTL2_4S(3, 11)
	VFMUL_S4(8, 4, 4)
	VFMUL_S4(9, 5, 5)
	VFMUL_S4(10, 6, 6)
	VFMUL_S4(11, 7, 7)
	VFCVTN_4H(4, 12)
	VFCVTN2_8H(5, 12)
	VFCVTN_4H(6, 13)
	VFCVTN2_8H(7, 13)
	VST1.P [V12.H8, V13.H8], 32(R2)
	SUB  $16, R3
	B    ai_fp16_pw_loop16

ai_fp16_pw_loop8:
	CMP  $8, R3
	BLT  ai_fp16_pw_scalar

	VLD1.P 16(R0), [V0.H8]
	VLD1.P 16(R1), [V2.H8]
	VFCVTL_4S(0, 4)
	VFCVTL2_4S(0, 5)
	VFCVTL_4S(2, 8)
	VFCVTL2_4S(2, 9)
	VFMUL_S4(8, 4, 4)
	VFMUL_S4(9, 5, 5)
	VFCVTN_4H(4, 12)
	VFCVTN2_8H(5, 12)
	VST1.P [V12.H8], 16(R2)
	SUB  $8, R3
	B    ai_fp16_pw_loop8

ai_fp16_pw_scalar:
	CBZ  R3, ai_fp16_pw_done

ai_fp16_pw_scalar_loop:
	MOVHU (R0), R4
	MOVHU (R1), R5
	VMOV R4, V0.H[0]
	VMOV R5, V2.H[0]
	VFCVTL_4S(0, 4)
	VFCVTL_4S(2, 8)
	VMOV V4.S[0], R6
	VMOV V8.S[0], R7
	FMOVS R6, F0
	FMOVS R7, F1
	FMULS F1, F0, F0
	AI_FP16_STORE_F32_AS_F16
	ADD  $2, R0
	ADD  $2, R1
	ADD  $2, R2
	SUB  $1, R3
	CBNZ R3, ai_fp16_pw_scalar_loop

ai_fp16_pw_done:
	RET

// func BeliefUpdateFloat16NEONAsm(likelihood, prior, output *uint16, count int)
TEXT ·BeliefUpdateFloat16NEONAsm(SB), NOSPLIT, $0-32
	MOVD likelihood+0(FP), R0
	MOVD prior+8(FP), R1
	MOVD output+16(FP), R2
	MOVD count+24(FP), R3
	CBZ  R3, ai_fp16_bu_done

	FMOVD $0, F14

ai_fp16_bu_loop16:
	CMP  $16, R3
	BLT  ai_fp16_bu_loop8

	VLD1.P 32(R0), [V0.H8, V1.H8]
	VLD1.P 32(R1), [V2.H8, V3.H8]
	VFCVTL_4S(0, 4)
	VFCVTL2_4S(0, 5)
	VFCVTL_4S(1, 6)
	VFCVTL2_4S(1, 7)
	VFCVTL_4S(2, 8)
	VFCVTL2_4S(2, 9)
	VFCVTL_4S(3, 10)
	VFCVTL2_4S(3, 11)
	VFMUL_S4(8, 4, 4)
	VFMUL_S4(9, 5, 5)
	VFMUL_S4(10, 6, 6)
	VFMUL_S4(11, 7, 7)
	AI_F32X4_TO_F64_ADD_F14(4)
	AI_F32X4_TO_F64_ADD_F14(5)
	AI_F32X4_TO_F64_ADD_F14(6)
	AI_F32X4_TO_F64_ADD_F14(7)
	VFCVTN_4H(4, 12)
	VFCVTN2_8H(5, 12)
	VFCVTN_4H(6, 13)
	VFCVTN2_8H(7, 13)
	VST1.P [V12.H8, V13.H8], 32(R2)
	SUB  $16, R3
	B    ai_fp16_bu_loop16

ai_fp16_bu_loop8:
	CMP  $8, R3
	BLT  ai_fp16_bu_scalar

	VLD1.P 16(R0), [V0.H8]
	VLD1.P 16(R1), [V2.H8]
	VFCVTL_4S(0, 4)
	VFCVTL2_4S(0, 5)
	VFCVTL_4S(2, 8)
	VFCVTL2_4S(2, 9)
	VFMUL_S4(8, 4, 4)
	VFMUL_S4(9, 5, 5)
	AI_F32X4_TO_F64_ADD_F14(4)
	AI_F32X4_TO_F64_ADD_F14(5)
	VFCVTN_4H(4, 12)
	VFCVTN2_8H(5, 12)
	VST1.P [V12.H8], 16(R2)
	SUB  $8, R3
	B    ai_fp16_bu_loop8

ai_fp16_bu_scalar:
	CBZ  R3, ai_fp16_bu_normalize

ai_fp16_bu_scalar_loop:
	MOVHU (R0), R4
	MOVHU (R1), R5
	VMOV R4, V0.H[0]
	VMOV R5, V2.H[0]
	VFCVTL_4S(0, 4)
	VFCVTL_4S(2, 8)
	VMOV V4.S[0], R6
	VMOV V8.S[0], R7
	FMOVS R6, F0
	FMOVS R7, F1
	FMULS F1, F0, F0
	FCVTSD F0, F6
	FADDD F6, F14, F14
	AI_FP16_STORE_F32_AS_F16
	ADD  $2, R0
	ADD  $2, R1
	ADD  $2, R2
	SUB  $1, R3
	CBNZ R3, ai_fp16_bu_scalar_loop

ai_fp16_bu_normalize:
	FMOVD $0, F15
	FCMPD F14, F15
	BEQ  ai_fp16_bu_done

	FMOVD $1.0, F3
	FDIVD F14, F3, F3
	FCVTDS F3, F3
	FMOVS F3, R6
	VMOV R6, V24.S[0]
	VDUP V24.S[0], V24.S4

	MOVD output+16(FP), R2
	MOVD count+24(FP), R3

ai_fp16_bu_scale_loop16:
	CMP  $16, R3
	BLT  ai_fp16_bu_scale_loop8

	VLD1 (R2), [V0.H8, V1.H8]
	VFCVTL_4S(0, 4)
	VFCVTL2_4S(0, 5)
	VFCVTL_4S(1, 6)
	VFCVTL2_4S(1, 7)
	VFMUL_S4(24, 4, 4)
	VFMUL_S4(24, 5, 5)
	VFMUL_S4(24, 6, 6)
	VFMUL_S4(24, 7, 7)
	VFCVTN_4H(4, 12)
	VFCVTN2_8H(5, 12)
	VFCVTN_4H(6, 13)
	VFCVTN2_8H(7, 13)
	VST1 [V12.H8, V13.H8], (R2)
	ADD  $32, R2
	SUB  $16, R3
	B    ai_fp16_bu_scale_loop16

ai_fp16_bu_scale_loop8:
	CMP  $8, R3
	BLT  ai_fp16_bu_scale_scalar

	VLD1 (R2), [V0.H8]
	VFCVTL_4S(0, 4)
	VFCVTL2_4S(0, 5)
	VFMUL_S4(24, 4, 4)
	VFMUL_S4(24, 5, 5)
	VFCVTN_4H(4, 12)
	VFCVTN2_8H(5, 12)
	VST1 [V12.H8], (R2)
	ADD  $16, R2
	SUB  $8, R3
	B    ai_fp16_bu_scale_loop8

ai_fp16_bu_scale_scalar:
	CBZ  R3, ai_fp16_bu_done

ai_fp16_bu_scale_scalar_loop:
	MOVHU (R2), R4
	VMOV R4, V0.H[0]
	VFCVTL_4S(0, 4)
	VMOV V4.S[0], R6
	FMOVS R6, F0
	FMULS F3, F0, F0
	AI_FP16_STORE_F32_AS_F16
	ADD  $2, R2
	SUB  $1, R3
	CBNZ R3, ai_fp16_bu_scale_scalar_loop

ai_fp16_bu_done:
	RET

// func FreeEnergyFloat16NEONAsm(likelihood, posterior, prior *uint16, count int) uint16
TEXT ·FreeEnergyFloat16NEONAsm(SB), NOSPLIT, $0-34
	MOVD likelihood+0(FP), R10
	MOVD posterior+8(FP), R11
	MOVD prior+16(FP), R12
	MOVD count+24(FP), R3

	MOVD $0x2b8cbccc, R17
	VMOV R17, V31.S[0]
	VDUP V31.S[0], V31.S4
	AI_LOAD_LOG_CONSTS
	VEOR V28.B16, V28.B16, V28.B16
	VEOR V29.B16, V29.B16, V29.B16
	FMOVD $0, F14
	FMOVD $0, F15
	CBZ  R3, ai_fp16_fe_store

ai_fp16_fe_loop8:
	CMP  $8, R3
	BLT  ai_fp16_fe_tail

	AI_FP16_FE_BLOCK8
	B    ai_fp16_fe_loop8

ai_fp16_fe_tail:
	CBZ  R3, ai_fp16_fe_store

ai_fp16_fe_tail_loop:
	AI_FP16_FE_TAIL_ONE
	CBNZ R3, ai_fp16_fe_tail_loop

ai_fp16_fe_store:
	AI_FP16_STORE_RESULT
	RET

// func ExpectedFreeEnergyFloat16NEONAsm(
//     predictedObs, preferredObs, predictedState *uint16,
//     obsCount, stateCount int,
// ) uint16
TEXT ·ExpectedFreeEnergyFloat16NEONAsm(SB), NOSPLIT, $0-42
	MOVD predictedObs+0(FP), R0
	MOVD preferredObs+8(FP), R1
	MOVD predictedState+16(FP), R2
	MOVD obsCount+24(FP), R3
	MOVD stateCount+32(FP), R4

	MOVD $0x2b8cbccc, R17
	VMOV R17, V31.S[0]
	VDUP V31.S[0], V31.S4
	AI_LOAD_LOG_CONSTS
	VEOR V28.B16, V28.B16, V28.B16
	VEOR V29.B16, V29.B16, V29.B16
	FMOVD $0, F14
	FMOVD $0, F15

ai_fp16_efe_obs_loop8:
	CMP  $8, R3
	BLT  ai_fp16_efe_obs_tail

	AI_FP16_EFE_OBS_BLOCK8
	B    ai_fp16_efe_obs_loop8

ai_fp16_efe_obs_tail:
	CBZ  R3, ai_fp16_efe_state_init

ai_fp16_efe_obs_tail_loop:
	AI_FP16_EFE_OBS_TAIL_ONE
	CBNZ R3, ai_fp16_efe_obs_tail_loop

ai_fp16_efe_state_init:
	MOVD predictedState+16(FP), R2
	MOVD stateCount+32(FP), R4

ai_fp16_efe_state_loop8:
	CMP  $8, R4
	BLT  ai_fp16_efe_state_tail

	AI_FP16_EFE_STATE_BLOCK8
	B    ai_fp16_efe_state_loop8

ai_fp16_efe_state_tail:
	CBZ  R4, ai_fp16_efe_store

ai_fp16_efe_state_tail_loop:
	AI_FP16_EFE_STATE_TAIL_ONE
	CBNZ R4, ai_fp16_efe_state_tail_loop

ai_fp16_efe_store:
	FADDP_D(28, 0)
	FADDP_D(29, 1)
	FADDD F14, F0, F0
	FADDD F15, F1, F1
	FADDD F1, F0, F0
	FCVTDS F0, F0
	FCVTSH F0, F0
	FMOVD F0, R6
	MOVH R6, ret+40(FP)
	RET
