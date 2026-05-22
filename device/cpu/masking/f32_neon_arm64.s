// SPDX-License-Identifier: Apache-2.0
// NEON float32 masking kernels: apply-mask add, causal mask, ALiBi bias.
#include "textflag.h"

#define VFADD_S4(m, n, d) WORD $(0x4E20D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_S4(m, n, d) WORD $(0x6E20DC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFSUB_S4(m, n, d) WORD $(0x4EA0D400 | ((m) << 16) | ((n) << 5) | (d))
#define VBSL_B16(m, n, d) WORD $(0x6E601C00 | ((m) << 16) | ((n) << 5) | (d))
#define VFCMGE_S4(m, n, d) WORD $(0x4EA0E400 | ((m) << 16) | ((n) << 5) | (d))

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

DATA maskIotaF32<>+0(SB)/4, $0.0
DATA maskIotaF32<>+4(SB)/4, $1.0
DATA maskIotaF32<>+8(SB)/4, $2.0
DATA maskIotaF32<>+12(SB)/4, $3.0
GLOBL maskIotaF32<>(SB), RODATA|NOPTR, $16

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

// func CausalMaskFloat32NEONAsm(output *float32, seqQ, seqK int)
TEXT ·CausalMaskFloat32NEONAsm(SB), NOSPLIT, $0-24
	MOVD output+0(FP), R0
	MOVD seqQ+8(FP), R16
	MOVD R16, R10
	MOVD seqK+16(FP), R5
	MOVD $maskZero<>(SB), R11

causal_row:
	CBZ  R10, causal_done

	MOVD R16, R7
	SUB  R10, R7, R7
	MOVD R5, R6

causal_col:
	CBZ  R6, causal_row_done

	MOVD R5, R4
	SUB  R6, R4, R8
	CMP  R8, R7
	BGT  causal_inf_cell

	FMOVS (R11), F0
	FMOVS F0, (R0)
	B    causal_col_next

causal_inf_cell:
	MOVD $0xFF800000, R12
	MOVW R12, (R0)

causal_col_next:
	ADD  $4, R0
	SUB  $1, R6
	B    causal_col

causal_row_done:
	SUB  $1, R10
	B    causal_row

causal_done:
	RET

// func ALiBiBiasFloat32NEONAsm(scores, slope, output *float32, seqQ, seqK int)
TEXT ·ALiBiBiasFloat32NEONAsm(SB), NOSPLIT, $0-40
	MOVD scores+0(FP), R0
	MOVD slope+8(FP), R1
	MOVD output+16(FP), R2
	MOVD seqQ+24(FP), R3
	MOVD seqK+32(FP), R4
	MOVD seqQ+24(FP), R16
	MOVD $maskIotaF32<>(SB), R10
	VLD1 (R10), [V14.S4]
	VEOR V9.B16, V9.B16, V9.B16

alibi_row:
	CBZ  R3, alibi_done

	MOVD R16, R5
	SUB  R3, R5, R5
	FMOVS (R1), F31
	VDUP  V31.S[0], V31.S4
	VMOV  R5, V13.S[0]
	VDUP  V13.S[0], V13.S4
	MOVD  $0, R6

alibi_col:
	MOVD R4, R7
	SUB  R6, R7, R7
	CBZ  R7, alibi_row_done

	CMP  $4, R7
	BLT  alibi_col_scalar_tail

	VLD1 (R0), [V0.S4]
	VMOV  R6, V12.S[0]
	VDUP  V12.S[0], V12.S4
	VFADD_S4(14, 12, 11)
	VFSUB_S4(11, 13, 10)
	VFCMGE_S4(10, 9, 8)
	VFMUL_S4(31, 10, 10)
	VFSUB_S4(10, 0, 1)
	VBSL_B16(1, 0, 8)
	VST1 [V8.S4], (R2)

	ADD  $16, R0
	ADD  $16, R2
	ADD  $4, R6
	B    alibi_col

alibi_col_scalar_tail:
	CBZ  R7, alibi_row_done

alibi_col_scalar_loop:
	FMOVS (R0), F0
	SUB   R5, R6, R8
	CMP   $0, R8
	BLT   alibi_keep_score

	SCVTFWS R8, F1
	FMULS F31, F1, F1
	FSUBS F1, F0, F0

alibi_keep_score:
	FMOVS F0, (R2)
	ADD  $4, R0
	ADD  $4, R2
	ADD  $1, R6
	SUB  $1, R7
	CBNZ R7, alibi_col_scalar_loop

alibi_row_done:
	SUB  $1, R3
	B    alibi_row

alibi_done:
	RET
