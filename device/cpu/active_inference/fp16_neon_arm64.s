// SPDX-License-Identifier: Apache-2.0
// NEON active inference float16 BeliefUpdate / PrecisionWeight kernels.
#include "textflag.h"

#define VFMUL_H8(m, n, d)  WORD $(0x6E401C00 | ((m) << 16) | ((n) << 5) | (d))
#define VFCVTL_4S(n, d)     WORD $(0x0E217800 | ((n) << 5) | (d))
#define VFCVTL2_4S(n, d)    WORD $(0x4E217800 | ((n) << 5) | (d))
#define VFCVTN_4H(n, d)     WORD $(0x0E216800 | ((n) << 5) | (d))
#define VFCVTN2_8H(n, d)    WORD $(0x4E216800 | ((n) << 5) | (d))
#define VFMUL_S4(m, n, d)   WORD $(0x6E20DC00 | ((m) << 16) | ((n) << 5) | (d))

#define AI_FP16_STORE_F32_AS_F16 \
	FCVTSH F0, F0 ;\
	FMOVD F0, R6 ;\
	MOVH R6, (R2)

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
	VFMUL_H8(2, 0, 16)
	VFMUL_H8(3, 1, 17)
	VST1.P [V16.H8, V17.H8], 32(R2)
	SUB  $16, R3
	B    ai_fp16_pw_loop16

ai_fp16_pw_loop8:
	CMP  $8, R3
	BLT  ai_fp16_pw_scalar

	VLD1.P 16(R0), [V0.H8]
	VLD1.P 16(R1), [V2.H8]
	VFMUL_H8(2, 0, 16)
	VST1.P [V16.H8], 16(R2)
	SUB  $8, R3
	B    ai_fp16_pw_loop8

ai_fp16_pw_scalar:
	CBZ  R3, ai_fp16_pw_done

ai_fp16_pw_scalar_loop:
	MOVHU (R0), R4
	MOVHU (R1), R5
	VMOV R4, V0.H[0]
	VMOV R5, V2.H[0]
	VFMUL_H8(2, 0, 16)
	VMOV V16.H[0], R6
	MOVH R6, (R2)
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

ai_fp16_bu_store_loop16:
	CMP  $16, R3
	BLT  ai_fp16_bu_store_loop8

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
	B    ai_fp16_bu_store_loop16

ai_fp16_bu_store_loop8:
	CMP  $8, R3
	BLT  ai_fp16_bu_store_scalar

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
	B    ai_fp16_bu_store_loop8

ai_fp16_bu_store_scalar:
	CBZ  R3, ai_fp16_bu_sum

ai_fp16_bu_store_scalar_loop:
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
	CBNZ R3, ai_fp16_bu_store_scalar_loop

ai_fp16_bu_sum:
	MOVD likelihood+0(FP), R0
	MOVD prior+8(FP), R1
	MOVD count+24(FP), R3
	FMOVD $0, F25

ai_fp16_bu_sum_loop:
	CBZ  R3, ai_fp16_bu_normalize

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
	FADDD F6, F25, F25
	ADD  $2, R0
	ADD  $2, R1
	SUB  $1, R3
	B    ai_fp16_bu_sum_loop

ai_fp16_bu_normalize:
	FMOVD $0, F15
	FCMPD F25, F15
	BEQ  ai_fp16_bu_done

	FMOVD $1.0, F3
	FDIVD F25, F3, F3
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
