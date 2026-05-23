// SPDX-License-Identifier: Apache-2.0
// NEON active inference bfloat16 BeliefUpdate / PrecisionWeight kernels.
#include "textflag.h"

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

#define AI_BF16_STORE_F32_AS_BF16 \
	FMOVS F0, R6 ;\
	LSR  $16, R6, R6 ;\
	MOVH R6, (R2)

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
TEXT ·BeliefUpdateBFloat16NEONAsm(SB), NOSPLIT, $0-32
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
	FMOVS F3, R6
	VMOV R6, V24.S[0]
	VDUP V24.S[0], V24.S4

	MOVD output+16(FP), R2
	MOVD count+24(FP), R3
	VEOR V30.B16, V30.B16, V30.B16

ai_bf16_bu_scale_loop16:
	CMP  $16, R3
	BLT  ai_bf16_bu_scale_loop8

	VLD1 (R2), [V0.H8, V1.H8]
	BF16_WIDEN_H8_TO_S4_LOW(V0.H8, V4.H8)
	BF16_WIDEN_H8_TO_S4_HIGH(V0.H8, V5.H8)
	BF16_WIDEN_H8_TO_S4_LOW(V1.H8, V6.H8)
	BF16_WIDEN_H8_TO_S4_HIGH(V1.H8, V7.H8)
	VFMUL_S4(24, 4, 4)
	VFMUL_S4(24, 5, 5)
	VFMUL_S4(24, 6, 6)
	VFMUL_S4(24, 7, 7)
	BF16_NARROW_S4_TO_H8(V4.H8, V5.H8, V12.H8)
	BF16_NARROW_S4_TO_H8(V6.H8, V7.H8, V13.H8)
	VST1 [V12.H8, V13.H8], (R2)
	ADD  $32, R2
	SUB  $16, R3
	B    ai_bf16_bu_scale_loop16

ai_bf16_bu_scale_loop8:
	CMP  $8, R3
	BLT  ai_bf16_bu_scale_scalar

	VLD1 (R2), [V0.H8]
	BF16_WIDEN_H8_TO_S4_LOW(V0.H8, V4.H8)
	BF16_WIDEN_H8_TO_S4_HIGH(V0.H8, V5.H8)
	VFMUL_S4(24, 4, 4)
	VFMUL_S4(24, 5, 5)
	BF16_NARROW_S4_TO_H8(V4.H8, V5.H8, V12.H8)
	VST1 [V12.H8], (R2)
	ADD  $16, R2
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
