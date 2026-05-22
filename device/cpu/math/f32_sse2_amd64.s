// SPDX-License-Identifier: Apache-2.0
// SSE2 float32 math kernels: inv_sqrt_dim_scale, logsumexp row parts, outer.
#include "textflag.h"

DATA mathOneF32<>+0(SB)/4, $0x3f800000
GLOBL mathOneF32<>(SB), RODATA|NOPTR, $4

DATA mathExpC<>+0(SB)/4, $1.4426950408889634
DATA mathExpC<>+4(SB)/4, $0.6931471805599453
DATA mathExpC<>+12(SB)/4, $0.00019841270
DATA mathExpC<>+16(SB)/4, $0.0013888889
DATA mathExpC<>+20(SB)/4, $0.008333334
DATA mathExpC<>+24(SB)/4, $0.041666667
DATA mathExpC<>+28(SB)/4, $0.16666667
DATA mathExpC<>+32(SB)/4, $0.5
DATA mathExpC<>+36(SB)/4, $1.0
DATA mathExpC<>+40(SB)/4, $1.0
GLOBL mathExpC<>(SB), RODATA|NOPTR, $44

DATA mathExpBias127<>+0(SB)/4, $127
GLOBL mathExpBias127<>(SB), RODATA|NOPTR, $4

DATA mathSoftmaxClamp<>+0(SB)/4, $-87.0
GLOBL mathSoftmaxClamp<>(SB), RODATA|NOPTR, $4

// func InvSqrtDimScaleFloat32SSE2Asm(out, input *float32, scale float32, count int)
TEXT ·InvSqrtDimScaleFloat32SSE2Asm(SB), NOSPLIT, $0-28
	MOVQ out+0(FP), DI
	MOVQ input+8(FP), SI
	MOVSS scale+16(FP), X15
	SHUFPS $0, X15, X15
	MOVQ count+20(FP), CX

inv_sqrt_sse2_w4:
	CMPQ CX, $4
	JL   inv_sqrt_sse2_tail

	MOVUPS (SI), X0
	MULPS  X15, X0
	MOVUPS X0, (DI)

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  inv_sqrt_sse2_w4

inv_sqrt_sse2_tail:
	TESTQ CX, CX
	JZ   inv_sqrt_sse2_done

inv_sqrt_sse2_scalar:
	MOVSS (SI), X0
	MULSS X15, X0
	MOVSS X0, (DI)
	ADDQ  $4, SI
	ADDQ  $4, DI
	DECQ  CX
	JNZ  inv_sqrt_sse2_scalar

inv_sqrt_sse2_done:
	RET

// func LogSumExpRowPartsFloat32SSE2Asm(row *float32, cols int, maximum, expSum *float32)
TEXT ·LogSumExpRowPartsFloat32SSE2Asm(SB), NOSPLIT, $32-32
	MOVQ row+0(FP), SI
	MOVQ cols+8(FP), CX
	TESTQ CX, CX
	JZ   lse_sse2_zero

	MOVSS (SI), X0
	SHUFPS $0, X0, X0
	ADDQ $4, SI
	DECQ CX

lse_sse2_max_w4:
	CMPQ CX, $4
	JL   lse_sse2_max_tail

	MOVUPS (SI), X1
	MAXPS  X1, X0

	ADDQ $16, SI
	SUBQ $4, CX
	JMP  lse_sse2_max_w4

lse_sse2_max_tail:
	TESTQ CX, CX
	JZ   lse_sse2_max_reduce

lse_sse2_max_scalar:
	MOVSS (SI), X1
	MAXSS X1, X0
	ADDQ  $4, SI
	DECQ  CX
	JNZ  lse_sse2_max_scalar

lse_sse2_max_reduce:
	MOVAPS X0, X1
	SHUFPS $0x4E, X0, X1
	MAXPS  X1, X0
	MOVAPS X0, X1
	SHUFPS $0xB1, X0, X1
	MAXPS  X1, X0
	MOVSS  X0, X6
	SHUFPS $0, X6, X6

	MOVQ row+0(FP), SI
	MOVQ cols+8(FP), CX

	MOVQ $mathExpC<>(SB), AX
	MOVSS (AX), X8
	SHUFPS $0, X8, X8
	MOVSS 4(AX), X9
	SHUFPS $0, X9, X9
	MOVSS 12(AX), X11
	SHUFPS $0, X11, X11
	MOVSS 16(AX), X12
	SHUFPS $0, X12, X12
	MOVSS 20(AX), X13
	SHUFPS $0, X13, X13
	MOVSS 24(AX), X14
	SHUFPS $0, X14, X14
	MOVSS 28(AX), X15
	SHUFPS $0, X15, X15
	MOVSS 32(AX), X0
	SHUFPS $0, X0, X0
	MOVAPS X0, lse_sse2_c8+0(SP)
	MOVSS 36(AX), X0
	SHUFPS $0, X0, X0
	MOVAPS X0, lse_sse2_c9+16(SP)
	MOVQ AX, R15
	MOVSS mathSoftmaxClamp<>(SB), X8
	SHUFPS $0, X8, X8
	MOVAPS X8, X4
	MOVSS mathOneF32<>(SB), X10
	SHUFPS $0, X10, X10
	XORPS X5, X5

lse_sse2_exp_w4:
	CMPQ CX, $4
	JL   lse_sse2_exp_tail

	MOVUPS (SI), X0
	SUBPS X6, X0
	DIVPS X10, X0
	MAXPS X4, X0
	MOVAPS X0, X1
	MULPS X8, X1
	ROUNDPS $8, X1, X1
	MOVAPS X1, X2
	MULPS X9, X2
	SUBPS X2, X0
	MOVAPS X11, X3
	MULPS X0, X11
	ADDPS X3, X11
	MOVAPS X12, X3
	MULPS X0, X12
	ADDPS X3, X12
	MOVAPS X13, X3
	MULPS X0, X13
	ADDPS X3, X13
	MOVAPS X14, X3
	MULPS X0, X14
	ADDPS X3, X14
	MOVAPS X15, X3
	MULPS X0, X15
	ADDPS X3, X15
	MOVAPS X15, X7
	MULPS X0, X7
	ADDPS lse_sse2_c8+0(SP), X7
	MULPS X0, X7
	ADDPS lse_sse2_c9+16(SP), X7
	CVTPS2PL X1, X1
	MOVD mathExpBias127<>(SB), X3
	PSHUFD $0, X3, X3
	PADDL X3, X1
	PSLLL $23, X1
	PADDL X1, X7
	ADDPS X7, X5

	ADDQ $16, SI
	SUBQ $4, CX
	JMP  lse_sse2_exp_w4

lse_sse2_exp_tail:
	TESTQ CX, CX
	JZ   lse_sse2_exp_reduce

	MOVSS 12(R15), X11
	MOVSS 16(R15), X12
	MOVSS 20(R15), X13
	MOVSS 24(R15), X14
	MOVSS 28(R15), X15
	MOVSS 32(R15), X0
	SHUFPS $0, X0, X0
	MOVAPS X0, lse_sse2_c8+0(SP)
	MOVSS 36(R15), X0
	SHUFPS $0, X0, X0
	MOVAPS X0, lse_sse2_c9+16(SP)

lse_sse2_exp_scalar:
	MOVSS (SI), X0
	SUBSS X6, X0
	DIVSS X10, X0
	MAXSS X4, X0
	MOVSS X0, X1
	MULSS X8, X1
	ROUNDSS $8, X1, X1
	MOVSS X1, X2
	MULSS X9, X2
	SUBSS X2, X0
	MOVSS 12(R15), X11
	MOVSS 16(R15), X12
	MOVSS 20(R15), X13
	MOVSS 24(R15), X14
	MOVSS 28(R15), X15
	MOVSS X11, X3
	MULSS X0, X11
	ADDSS X3, X11
	MOVSS X12, X3
	MULSS X0, X12
	ADDSS X3, X12
	MOVSS X13, X3
	MULSS X0, X13
	ADDSS X3, X13
	MOVSS X14, X3
	MULSS X0, X14
	ADDSS X3, X14
	MOVSS X15, X3
	MULSS X0, X15
	ADDSS X3, X15
	MOVSS X15, X7
	MULSS X0, X7
	ADDSS lse_sse2_c8+0(SP), X7
	MULSS X0, X7
	ADDSS lse_sse2_c9+16(SP), X7
	XORPS X2, X2
	MOVSS X1, X2
	CVTPS2PL X2, X2
	MOVD mathExpBias127<>(SB), X3
	PSHUFD $0, X3, X3
	PADDL X3, X2
	PSLLL $23, X2
	PADDL X2, X7
	ADDSS X7, X5
	ADDQ  $4, SI
	DECQ  CX
	JNZ  lse_sse2_exp_scalar

lse_sse2_exp_reduce:
	MOVAPS X5, X0
	SHUFPS $1, X5, X5
	ADDPS X5, X0
	MOVAPS X0, X5
	SHUFPS $2, X0, X0
	ADDPS X5, X0

	MOVQ maximum+16(FP), DI
	MOVQ expSum+24(FP), SI
	MOVSS X6, (DI)
	MOVSS X0, (SI)
	RET

lse_sse2_zero:
	MOVQ maximum+16(FP), DI
	MOVQ expSum+24(FP), SI
	XORPS X0, X0
	MOVSS X0, (DI)
	MOVSS X0, (SI)
	RET

// func OuterFloat32SSE2Asm(out, left, right *float32, leftCount, rightCount int)
TEXT ·OuterFloat32SSE2Asm(SB), NOSPLIT, $0-40
	MOVQ out+0(FP), DI
	MOVQ left+8(FP), SI
	MOVQ right+16(FP), R8
	MOVQ leftCount+24(FP), R9
	MOVQ rightCount+32(FP), R10

	MOVQ R10, R11
	SHLQ $2, R11

outer_sse2_row:
	TESTQ R9, R9
	JZ   outer_sse2_done

	MOVSS (SI), X0
	SHUFPS $0, X0, X0
	MOVQ R8, BX
	MOVQ R10, CX

outer_sse2_col_w4:
	CMPQ CX, $4
	JL   outer_sse2_col_tail

	MOVUPS (BX), X1
	MULPS  X0, X1
	MOVUPS X1, (DI)

	ADDQ $16, BX
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  outer_sse2_col_w4

outer_sse2_col_tail:
	TESTQ CX, CX
	JZ   outer_sse2_next_row

outer_sse2_col_scalar:
	MOVSS (BX), X1
	MULSS X0, X1
	MOVSS X1, (DI)
	ADDQ  $4, BX
	ADDQ  $4, DI
	DECQ  CX
	JNZ  outer_sse2_col_scalar

outer_sse2_next_row:
	ADDQ $4, SI
	ADDQ R11, DI
	DECQ R9
	JMP  outer_sse2_row

outer_sse2_done:
	RET
