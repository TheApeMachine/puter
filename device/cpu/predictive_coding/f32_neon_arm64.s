// SPDX-License-Identifier: Apache-2.0
// NEON predictive coding float32 kernels.
#include "textflag.h"

#define VFSUB_S4(m, n, d)  WORD $(0x4EA0D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_S4(m, n, d)  WORD $(0x6E20DC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFADD_S4(m, n, d)  WORD $(0x4E20D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMLA_D2(m, n, d)  WORD $(0x4E60CC00 | ((m) << 16) | ((n) << 5) | (d))
#define FCVTL_2D(n, d)     WORD $(0x0E617800 | ((n) << 5) | (d))
#define FCVTL2_2D(n, d)    WORD $(0x4E617800 | ((n) << 5) | (d))
#define VFADD_D2(m, n, d)  WORD $(0x4E60D400 | ((m) << 16) | ((n) << 5) | (d))
#define FADDP_D(n, d)      WORD $(0x7E70D800 | ((n) << 5) | (d))

// func PCPredictionErrorFloat32NEONAsm(observed, predicted, output *float32, count int)
TEXT ·PCPredictionErrorFloat32NEONAsm(SB), NOSPLIT, $0-28
	MOVD observed+0(FP), R0
	MOVD predicted+8(FP), R1
	MOVD output+16(FP), R2
	MOVD count+24(FP), R3

pc_pe_loop4:
	CMP  $4, R3
	BLT  pc_pe_scalar

	VLD1 (R0), [V0.S4]
	VLD1 (R1), [V1.S4]
	VFSUB_S4(1, 0, 0)
	VST1 [V0.S4], (R2)
	ADD  $16, R0
	ADD  $16, R1
	ADD  $16, R2
	SUB  $4, R3
	B    pc_pe_loop4

pc_pe_scalar:
	CBZ  R3, pc_pe_done

pc_pe_scalar_loop:
	FMOVS (R0), F0
	FMOVS (R1), F1
	FSUBS F1, F0, F0
	FMOVS F0, (R2)
	ADD  $4, R0
	ADD  $4, R1
	ADD  $4, R2
	SUB  $1, R3
	CBNZ R3, pc_pe_scalar_loop

pc_pe_done:
	RET

// func PCPredictionFloat32NEONAsm(weights, representation, output *float32, outDim, inDim int)
TEXT ·PCPredictionFloat32NEONAsm(SB), NOSPLIT, $0-36
	MOVD weights+0(FP), R11
	MOVD representation+8(FP), R12
	MOVD output+16(FP), R13
	MOVD outDim+24(FP), R9
	MOVD inDim+32(FP), R8

pc_pred_row:
	CBZ  R9, pc_pred_done

	VEOR V16.B16, V16.B16, V16.B16
	FMOVD $0, F14
	MOVD R11, R0
	MOVD R12, R1
	MOVD R8, R2

pc_pred_dot_loop4:
	CMP  $4, R2
	BLT  pc_pred_dot_tail

	VLD1.P 16(R0), [V0.S4]
	VLD1.P 16(R1), [V4.S4]
	FCVTL_2D(0, 8)
	FCVTL2_2D(0, 9)
	FCVTL_2D(4, 10)
	FCVTL2_2D(4, 11)
	VFMLA_D2(10, 8, 16)
	VFMLA_D2(11, 9, 16)
	SUB  $4, R2
	B    pc_pred_dot_loop4

pc_pred_dot_tail:
	CBZ  R2, pc_pred_dot_reduce

pc_pred_dot_tail_loop:
	FMOVS (R0), F0
	FMOVS (R1), F1
	FCVTSD F0, F0
	FCVTSD F1, F1
	FMULD F1, F0, F0
	FADDD F0, F14, F14
	ADD  $4, R0
	ADD  $4, R1
	SUB  $1, R2
	CBNZ R2, pc_pred_dot_tail_loop

pc_pred_dot_reduce:
	FADDP_D(16, 0)
	FADDD F0, F14, F0
	FCVTDS F0, F0
	FMOVS F0, (R13)
	ADD  $4, R13
	MOVD R8, R3
	LSL  $2, R3, R3
	ADD  R3, R11
	SUB  $1, R9
	B    pc_pred_row

pc_pred_done:
	RET

// func PCUpdateRepresentationFloat32NEONAsm(
//     weights, representation, predictionError, output *float32,
//     learningRate float32, outDim, inDim int,
// )
TEXT ·PCUpdateRepresentationFloat32NEONAsm(SB), NOSPLIT, $0-56
	MOVD weights+0(FP), R11
	MOVD representation+8(FP), R12
	MOVD predictionError+16(FP), R10
	MOVD output+24(FP), R13
	FMOVS learningRate+32(FP), F15
	MOVD outDim+40(FP), R9
	MOVD inDim+48(FP), R8

	MOVD R12, R0
	MOVD R13, R1
	MOVD R8, R2

pc_ur_copy_loop4:
	CMP  $4, R2
	BLT  pc_ur_copy_scalar

	VLD1 (R0), [V0.S4]
	VST1 [V0.S4], (R1)
	ADD  $16, R0
	ADD  $16, R1
	SUB  $4, R2
	B    pc_ur_copy_loop4

pc_ur_copy_scalar:
	CBZ  R2, pc_ur_rows

pc_ur_copy_scalar_loop:
	FMOVS (R0), F0
	FMOVS F0, (R1)
	ADD  $4, R0
	ADD  $4, R1
	SUB  $1, R2
	CBNZ R2, pc_ur_copy_scalar_loop

pc_ur_rows:
pc_ur_row:
	CBZ  R9, pc_ur_done

	FMOVS (R10), F0
	FCVTSD F15, F16
	FCVTSD F0, F0
	FMULD F0, F16, F16
	MOVD R11, R0
	MOVD R13, R1
	MOVD R8, R2

pc_ur_loop4:
	CMP  $4, R2
	BLT  pc_ur_scalar

	FMOVS (R0), F1
	FCVTSD F1, F1
	FMULD F16, F1, F1
	FCVTDS F1, F1
	FMOVS (R1), F2
	FADDS F1, F2, F2
	FMOVS F2, (R1)
	ADD  $4, R0
	ADD  $4, R1

	FMOVS (R0), F1
	FCVTSD F1, F1
	FMULD F16, F1, F1
	FCVTDS F1, F1
	FMOVS (R1), F2
	FADDS F1, F2, F2
	FMOVS F2, (R1)
	ADD  $4, R0
	ADD  $4, R1

	FMOVS (R0), F1
	FCVTSD F1, F1
	FMULD F16, F1, F1
	FCVTDS F1, F1
	FMOVS (R1), F2
	FADDS F1, F2, F2
	FMOVS F2, (R1)
	ADD  $4, R0
	ADD  $4, R1

	FMOVS (R0), F1
	FCVTSD F1, F1
	FMULD F16, F1, F1
	FCVTDS F1, F1
	FMOVS (R1), F2
	FADDS F1, F2, F2
	FMOVS F2, (R1)
	ADD  $4, R0
	ADD  $4, R1

	SUB  $4, R2
	B    pc_ur_loop4

pc_ur_scalar:
	CBZ  R2, pc_ur_next_row

pc_ur_scalar_loop:
	FMOVS (R0), F1
	FCVTSD F1, F1
	FMULD F16, F1, F1
	FCVTDS F1, F1
	FMOVS (R1), F2
	FADDS F1, F2, F2
	FMOVS F2, (R1)
	ADD  $4, R0
	ADD  $4, R1
	SUB  $1, R2
	CBNZ R2, pc_ur_scalar_loop

pc_ur_next_row:
	ADD  $4, R10
	MOVD R8, R3
	LSL  $2, R3, R3
	ADD  R3, R11
	SUB  $1, R9
	B    pc_ur_row

pc_ur_done:
	RET

// func PCUpdateWeightsFloat32NEONAsm(
//     weights, representation, predictionError, output *float32,
//     learningRate float32, outDim, inDim int,
// )
TEXT ·PCUpdateWeightsFloat32NEONAsm(SB), NOSPLIT, $0-56
	MOVD weights+0(FP), R11
	MOVD representation+8(FP), R12
	MOVD predictionError+16(FP), R10
	MOVD output+24(FP), R13
	FMOVS learningRate+32(FP), F15
	MOVD outDim+40(FP), R9
	MOVD inDim+48(FP), R8

	MOVD R11, R0
	MOVD R13, R1
	MOVD R9, R2
	MUL  R8, R2, R2

pc_uw_copy_loop4:
	CMP  $4, R2
	BLT  pc_uw_copy_scalar

	VLD1 (R0), [V0.S4]
	VST1 [V0.S4], (R1)
	ADD  $16, R0
	ADD  $16, R1
	SUB  $4, R2
	B    pc_uw_copy_loop4

pc_uw_copy_scalar:
	CBZ  R2, pc_uw_rows

pc_uw_copy_scalar_loop:
	FMOVS (R0), F0
	FMOVS F0, (R1)
	ADD  $4, R0
	ADD  $4, R1
	SUB  $1, R2
	CBNZ R2, pc_uw_copy_scalar_loop

pc_uw_rows:
pc_uw_row:
	CBZ  R9, pc_uw_done

	FMOVS (R10), F0
	FCVTSD F15, F16
	FCVTSD F0, F0
	FMULD F0, F16, F16
	MOVD R11, R0
	MOVD R12, R1
	MOVD R13, R2
	MOVD R8, R3

pc_uw_loop4:
	CMP  $4, R3
	BLT  pc_uw_scalar

	FMOVS (R1), F1
	FCVTSD F1, F1
	FMULD F16, F1, F1
	FCVTDS F1, F1
	FMOVS (R2), F4
	FADDS F1, F4, F4
	FMOVS F4, (R2)
	ADD  $4, R0
	ADD  $4, R1
	ADD  $4, R2

	FMOVS (R1), F1
	FCVTSD F1, F1
	FMULD F16, F1, F1
	FCVTDS F1, F1
	FMOVS (R2), F4
	FADDS F1, F4, F4
	FMOVS F4, (R2)
	ADD  $4, R0
	ADD  $4, R1
	ADD  $4, R2

	FMOVS (R1), F1
	FCVTSD F1, F1
	FMULD F16, F1, F1
	FCVTDS F1, F1
	FMOVS (R2), F4
	FADDS F1, F4, F4
	FMOVS F4, (R2)
	ADD  $4, R0
	ADD  $4, R1
	ADD  $4, R2

	FMOVS (R1), F1
	FCVTSD F1, F1
	FMULD F16, F1, F1
	FCVTDS F1, F1
	FMOVS (R2), F4
	FADDS F1, F4, F4
	FMOVS F4, (R2)
	ADD  $4, R0
	ADD  $4, R1
	ADD  $4, R2

	SUB  $4, R3
	B    pc_uw_loop4

pc_uw_scalar:
	CBZ  R3, pc_uw_next_row

pc_uw_scalar_loop:
	FMOVS (R1), F1
	FCVTSD F1, F1
	FMULD F16, F1, F1
	FCVTDS F1, F1
	FMOVS (R2), F4
	FADDS F1, F4, F4
	FMOVS F4, (R2)
	ADD  $4, R0
	ADD  $4, R1
	ADD  $4, R2
	SUB  $1, R3
	CBNZ R3, pc_uw_scalar_loop

pc_uw_next_row:
	ADD  $4, R10
	MOVD R8, R4
	LSL  $2, R4, R4
	ADD  R4, R11
	ADD  R4, R13
	SUB  $1, R9
	B    pc_uw_row

pc_uw_done:
	RET
