// SPDX-License-Identifier: Apache-2.0
// AVX2 float32 math kernels: inv_sqrt_dim_scale, logsumexp row parts, outer.
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

// func InvSqrtDimScaleFloat32AVX2Asm(out, input *float32, scale float32, count int)
TEXT ·InvSqrtDimScaleFloat32AVX2Asm(SB), NOSPLIT, $0-28
	MOVQ out+0(FP), DI
	MOVQ input+8(FP), SI
	MOVSS scale+16(FP), X15
	VBROADCASTSS X15, Y15
	MOVQ count+20(FP), CX

inv_sqrt_avx2_w8:
	CMPQ CX, $8
	JL   inv_sqrt_avx2_w4

	VMOVUPS (SI), Y0
	VMULPS  Y15, Y0, Y0
	VMOVUPS Y0, (DI)

	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $8, CX
	JMP  inv_sqrt_avx2_w8

inv_sqrt_avx2_w4:
	CMPQ CX, $4
	JL   inv_sqrt_avx2_tail

	VMOVUPS (SI), X0
	VMULPS  X15, X0, X0
	VMOVUPS X0, (DI)

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  inv_sqrt_avx2_w4

inv_sqrt_avx2_tail:
	TESTQ CX, CX
	JZ   inv_sqrt_avx2_done

inv_sqrt_avx2_scalar:
	MOVSS (SI), X0
	VMULSS X15, X0, X0
	MOVSS X0, (DI)
	ADDQ  $4, SI
	ADDQ  $4, DI
	DECQ  CX
	JNZ  inv_sqrt_avx2_scalar

inv_sqrt_avx2_done:
	RET

// func LogSumExpRowPartsFloat32AVX2Asm(row *float32, cols int, maximum, expSum *float32)
TEXT ·LogSumExpRowPartsFloat32AVX2Asm(SB), NOSPLIT, $0-32
	MOVQ row+0(FP), SI
	MOVQ cols+8(FP), CX
	TESTQ CX, CX
	JZ   lse_avx2_zero

	MOVSS (SI), X0
	VBROADCASTSS X0, Y0
	ADDQ $4, SI
	DECQ CX

lse_avx2_max_w8:
	CMPQ CX, $8
	JL   lse_avx2_max_w4

	VMOVUPS (SI), Y1
	VMAXPS  Y1, Y0, Y0

	ADDQ $32, SI
	SUBQ $8, CX
	JMP  lse_avx2_max_w8

lse_avx2_max_w4:
	CMPQ CX, $4
	JL   lse_avx2_max_reduce

	VMOVUPS (SI), Y1
	VMAXPS  Y1, Y0, Y0

	ADDQ $16, SI
	SUBQ $4, CX
	JMP  lse_avx2_max_w4

lse_avx2_max_reduce:
	VEXTRACTF128 $0, Y0, X0
	VEXTRACTF128 $1, Y0, X1
	VMAXPS  X1, X0, X0
	VHADDPS X0, X0, X0
	VHADDPS X0, X0, X0

	TESTQ CX, CX
	JZ   lse_avx2_max_broadcast

lse_avx2_max_scalar:
	MOVSS (SI), X1
	MAXSS X1, X0
	ADDQ  $4, SI
	DECQ  CX
	JNZ  lse_avx2_max_scalar

lse_avx2_max_broadcast:
	VBROADCASTSS X0, Y6

	MOVQ row+0(FP), SI
	MOVQ cols+8(FP), CX

	MOVQ $mathExpC<>(SB), AX
	VMOVSS (AX), X8
	VBROADCASTSS X8, Y8
	VMOVSS 4(AX), X9
	VBROADCASTSS X9, Y9
	VMOVSS 12(AX), X11
	VBROADCASTSS X11, Y11
	VMOVSS 16(AX), X12
	VBROADCASTSS X12, Y12
	VMOVSS 20(AX), X13
	VBROADCASTSS X13, Y13
	VMOVSS 24(AX), X14
	VBROADCASTSS X14, Y14
	VMOVSS 28(AX), X15
	VBROADCASTSS X15, Y15
	VMOVSS 32(AX), X16
	VBROADCASTSS X16, Y16
	VMOVSS 36(AX), X17
	VBROADCASTSS X17, Y17
	MOVQ AX, R15
	MOVD mathExpBias127<>(SB), R14
	VMOVSS mathSoftmaxClamp<>(SB), X8
	VBROADCASTSS X8, Y4
	VBROADCASTSS mathOneF32<>(SB), Y10
	VXORPS Y5, Y5, Y5

lse_avx2_exp_w8:
	CMPQ CX, $8
	JL   lse_avx2_exp_w4

	VMOVUPS (SI), Y0
	VSUBPS Y6, Y0, Y0
	VDIVPS Y10, Y0, Y0
	VMAXPS Y4, Y0, Y0
	VMULPS Y8, Y0, Y1
	VROUNDPS $8, Y1, Y1
	VMULPS Y1, Y9, Y2
	VSUBPS Y2, Y0, Y0
	VMOVAPS Y11, Y3
	VFMADD213PS Y3, Y0, Y11
	VMOVAPS Y12, Y3
	VFMADD213PS Y3, Y0, Y12
	VMOVAPS Y13, Y3
	VFMADD213PS Y3, Y0, Y13
	VMOVAPS Y14, Y3
	VFMADD213PS Y3, Y0, Y14
	VMOVAPS Y15, Y3
	VFMADD213PS Y3, Y0, Y15
	VMOVAPS Y16, Y3
	VFMADD213PS Y3, Y0, Y16
	VMOVAPS Y17, Y7
	VFMADD213PS Y7, Y0, Y17
	VCVTPS2DQ Y1, Y1
	MOVD R14, X3
	VPBROADCASTD X3, Y3
	VPADDD Y3, Y1, Y1
	VPSLLD $23, Y1, Y1
	VPADDD Y1, Y7, Y7
	VADDPS Y5, Y7, Y5

	ADDQ $32, SI
	SUBQ $8, CX
	JMP  lse_avx2_exp_w8

lse_avx2_exp_w4:
	CMPQ CX, $4
	JL   lse_avx2_exp_tail

	VMOVUPS (SI), X0
	VSUBPS X6, X0, X0
	VDIVPS X10, X0, X0
	VMAXPS X4, X0, X0
	VMULPS X8, X0, X1
	VROUNDPS $8, X1, X1
	VMULPS X1, X9, X2
	VSUBPS X2, X0, X0
	VMOVAPS X11, X3
	VFMADD213PS X3, X0, X11
	VMOVAPS X12, X3
	VFMADD213PS X3, X0, X12
	VMOVAPS X13, X3
	VFMADD213PS X3, X0, X13
	VMOVAPS X14, X3
	VFMADD213PS X3, X0, X14
	VMOVAPS X15, X3
	VFMADD213PS X3, X0, X15
	VMOVAPS X16, X3
	VFMADD213PS X3, X0, X16
	VMOVAPS X17, X7
	VFMADD213PS X7, X0, X17
	VCVTPS2DQ X1, X1
	MOVD R14, X3
	VPBROADCASTD X3, X3
	VPADDD X3, X1, X1
	VPSLLD $23, X1, X1
	VPADDD X1, X7, X7
	VADDPS X5, X7, X5

	ADDQ $16, SI
	SUBQ $4, CX
	JMP  lse_avx2_exp_w4

lse_avx2_exp_tail:
	TESTQ CX, CX
	JZ   lse_avx2_exp_reduce

	VMOVSS 32(R15), X16
	VBROADCASTSS X16, Y16
	VMOVSS 36(R15), X17
	VBROADCASTSS X17, Y17

lse_avx2_exp_scalar:
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
	VMOVSS 12(R15), X11
	VMOVSS 16(R15), X12
	VMOVSS 20(R15), X13
	VMOVSS 24(R15), X14
	VMOVSS 28(R15), X15
	MOVAPS X11, X3
	VFMADD213SS X3, X0, X11
	MOVAPS X12, X3
	VFMADD213SS X3, X0, X12
	MOVAPS X13, X3
	VFMADD213SS X3, X0, X13
	MOVAPS X14, X3
	VFMADD213SS X3, X0, X14
	MOVAPS X15, X3
	VFMADD213SS X3, X0, X15
	MOVSS 32(R15), X3
	MOVSS X3, X14
	VFMADD213SS X3, X0, X14
	MOVSS 36(R15), X7
	VFMADD213SS X7, X0, X7
	XORPS X2, X2
	MOVSS X1, X2
	VCVTPS2DQ X2, X2
	MOVD R14, X3
	PSHUFD $0, X3, X3
	PADDD X3, X2
	VPSLLD $23, X2, X2
	PADDD X2, X7
	ADDSS X7, X5
	ADDQ  $4, SI
	DECQ  CX
	JNZ  lse_avx2_exp_scalar

lse_avx2_exp_reduce:
	VHADDPS Y5, Y5, Y5
	VHADDPS Y5, Y5, Y5
	VEXTRACTF128 $0, Y5, X0

	MOVQ maximum+16(FP), DI
	MOVQ expSum+24(FP), SI
	MOVSS X6, (DI)
	MOVSS X0, (SI)
	RET

lse_avx2_zero:
	MOVQ maximum+16(FP), DI
	MOVQ expSum+24(FP), SI
	XORPS X0, X0
	MOVSS X0, (DI)
	MOVSS X0, (SI)
	RET

// func OuterFloat32AVX2Asm(out, left, right *float32, leftCount, rightCount int)
TEXT ·OuterFloat32AVX2Asm(SB), NOSPLIT, $0-40
	MOVQ out+0(FP), DI
	MOVQ left+8(FP), SI
	MOVQ right+16(FP), R8
	MOVQ leftCount+24(FP), R9
	MOVQ rightCount+32(FP), R10

	MOVQ R10, R11
	SHLQ $2, R11

outer_avx2_row:
	TESTQ R9, R9
	JZ   outer_avx2_done

	MOVSS (SI), X0
	VBROADCASTSS X0, Y0
	MOVQ R8, BX
	MOVQ R10, CX

outer_avx2_col_w8:
	CMPQ CX, $8
	JL   outer_avx2_col_w4

	VMOVUPS (BX), Y1
	VMULPS  Y0, Y1, Y1
	VMOVUPS Y1, (DI)

	ADDQ $32, BX
	ADDQ $32, DI
	SUBQ $8, CX
	JMP  outer_avx2_col_w8

outer_avx2_col_w4:
	CMPQ CX, $4
	JL   outer_avx2_col_tail

	VMOVUPS (BX), X1
	VMULPS  X0, X1, X1
	VMOVUPS X1, (DI)

	ADDQ $16, BX
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  outer_avx2_col_w4

outer_avx2_col_tail:
	TESTQ CX, CX
	JZ   outer_avx2_next_row

outer_avx2_col_scalar:
	MOVSS (BX), X1
	VMULSS X0, X1, X1
	MOVSS X1, (DI)
	ADDQ  $4, BX
	ADDQ  $4, DI
	DECQ  CX
	JNZ  outer_avx2_col_scalar

outer_avx2_next_row:
	ADDQ $4, SI
	ADDQ R11, DI
	DECQ R9
	JMP  outer_avx2_row

outer_avx2_done:
	RET
