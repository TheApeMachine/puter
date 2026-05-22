// SPDX-License-Identifier: Apache-2.0
// NEON float32 masking kernels: apply-mask add, causal mask, ALiBi bias.
#include "textflag.h"

#define VFADD_S4(m, n, d) WORD $(0x4E20D400 | ((m) << 16) | ((n) << 5) | (d))

DATA maskZero<>+0(SB)/4, $0.0
DATA maskZero<>+4(SB)/4, $0.0
DATA maskZero<>+8(SB)/4, $0.0
DATA maskZero<>+12(SB)/4, $0.0
GLOBL maskZero<>(SB), RODATA|NOPTR, $16

DATA maskNegInf<>+0(SB)/4, $0xFF800000
DATA maskNegInf<>+4(SB)/4, $0xFF800000
DATA maskNegInf<>+8(SB)/4, $0xFF800000
DATA maskNegInf<>+12(SB)/4, $0xFF800000
GLOBL maskNegInf<>(SB), RODATA|NOPTR, $16

// func ApplyMaskFloat32NEONAsm(input, mask, output *float32, count int)
TEXT ·ApplyMaskFloat32NEONAsm(SB), NOSPLIT, $0-32
	MOVD input+0(FP), R0
	MOVD mask+8(FP), R1
	MOVD output+16(FP), R2
	MOVD count+24(FP), R3

mask_apply_loop16:
	CMP  $16, R3
	BLT  mask_apply_loop4

	VLD1 (R0), [V0.S4, V1.S4, V2.S4, V3.S4]
	VLD1 (R1), [V4.S4, V5.S4, V6.S4, V7.S4]
	VFADD_S4(4, 0, 0)
	VFADD_S4(5, 1, 1)
	VFADD_S4(6, 2, 2)
	VFADD_S4(7, 3, 3)
	VST1 [V0.S4, V1.S4, V2.S4, V3.S4], (R2)

	ADD  $64, R0
	ADD  $64, R1
	ADD  $64, R2
	SUB  $16, R3
	B    mask_apply_loop16

mask_apply_loop4:
	CMP  $4, R3
	BLT  mask_apply_scalar_tail

	VLD1 (R0), [V0.S4]
	VLD1 (R1), [V4.S4]
	VFADD_S4(4, 0, 0)
	VST1 [V0.S4], (R2)

	ADD  $16, R0
	ADD  $16, R1
	ADD  $16, R2
	SUB  $4, R3
	B    mask_apply_loop4

mask_apply_scalar_tail:
	CBZ  R3, mask_apply_done

mask_apply_scalar_loop:
	FMOVS (R0), F0
	FMOVS (R1), F1
	FADDS F1, F0, F0
	FMOVS F0, (R2)
	ADD  $4, R0
	ADD  $4, R1
	ADD  $4, R2
	SUB  $1, R3
	CBNZ R3, mask_apply_scalar_loop

mask_apply_done:
	RET

// func causalMaskFloat32NEONFillAsm(rowOutput *float32, zeroCount, infCount int)
TEXT ·causalMaskFloat32NEONFillAsm(SB), NOSPLIT, $0-24
	MOVD rowOutput+0(FP), R0
	MOVD 16(RSP), R12
	MOVD 24(RSP), R11

causal_fill_zero_w4:
	CMP  $4, R12
	BLT  causal_fill_zero_tail

	MOVD $maskZero<>(SB), R3
	VLD1 (R3), [V0.S4]
	VST1 [V0.S4], (R0)
	ADD  $16, R0
	SUB  $4, R12
	B    causal_fill_zero_w4

causal_fill_zero_tail:
	CBZ  R12, causal_fill_zero_done

causal_fill_zero_scalar:
	FMOVS maskZero<>(SB), F0
	FMOVS F0, (R0)
	ADD  $4, R0
	SUB  $1, R12
	CBNZ R12, causal_fill_zero_scalar

causal_fill_zero_done:
	MOVD 24(RSP), R12

causal_fill_inf_w4:
	CMP  $4, R12
	BLT  causal_fill_inf_tail

	MOVD $maskNegInf<>(SB), R3
	VLD1 (R3), [V0.S4]
	VST1 [V0.S4], (R0)
	ADD  $16, R0
	SUB  $4, R12
	B    causal_fill_inf_w4

causal_fill_inf_tail:
	CBZ  R12, causal_fill_done

causal_fill_inf_scalar:
	FMOVS maskNegInf<>(SB), F0
	FMOVS F0, (R0)
	ADD  $4, R0
	SUB  $1, R12
	CBNZ R12, causal_fill_inf_scalar

causal_fill_done:
	RET

// func alibiBiasFloat32NEONElemAsm(score, slope, output *float32, distance int)
TEXT ·alibiBiasFloat32NEONElemAsm(SB), NOSPLIT, $0-32
	MOVD score+0(FP), R0
	MOVD slope+8(FP), R1
	MOVD output+16(FP), R2
	MOVD 32(RSP), R12

	FMOVS (R0), F0
	CMP   $0, R12
	BLT   alibi_elem_keep_score

	FMOVS (R1), F31
	SCVTFWS R12, F1
	FMULS F31, F1, F1
	FSUBS F1, F0, F0

alibi_elem_keep_score:
	FMOVS F0, (R2)
	RET
