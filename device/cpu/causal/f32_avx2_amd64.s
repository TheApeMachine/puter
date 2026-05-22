// SPDX-License-Identifier: Apache-2.0
// AVX2 float32 causal kernels: CATE subtract, counterfactual, strided dot.
#include "textflag.h"

// func CateFloat32AVX2Asm(treated, control, out *float32, count int)
TEXT ·CateFloat32AVX2Asm(SB), NOSPLIT, $0-32
	MOVQ treated+0(FP), DI
	MOVQ control+8(FP), SI
	MOVQ out+16(FP), R8
	MOVQ count+24(FP), CX

cate_avx2_w8:
	CMPQ CX, $8
	JL   cate_avx2_w4

	VMOVUPS (DI), Y0
	VMOVUPS (SI), Y1
	VSUBPS  Y1, Y0, Y0
	VMOVUPS Y0, (R8)

	ADDQ $32, DI
	ADDQ $32, SI
	ADDQ $32, R8
	SUBQ $8, CX
	JMP  cate_avx2_w8

cate_avx2_w4:
	CMPQ CX, $4
	JL   cate_avx2_tail

	VMOVUPS (DI), X0
	VMOVUPS (SI), X1
	VSUBPS  X1, X0, X0
	VMOVUPS X0, (R8)

	ADDQ $16, DI
	ADDQ $16, SI
	ADDQ $16, R8
	SUBQ $4, CX
	JMP  cate_avx2_w4

cate_avx2_tail:
	TESTQ CX, CX
	JZ   cate_avx2_done

cate_avx2_scalar:
	MOVSS (DI), X0
	SUBSS (SI), X0
	MOVSS X0, (R8)
	ADDQ  $4, DI
	ADDQ  $4, SI
	ADDQ  $4, R8
	DECQ  CX
	JNZ   cate_avx2_scalar

cate_avx2_done:
	RET

// func CounterfactualFloat32AVX2Asm(out, observedY, observedX, counterfactualX *float32, slope float32, count int)
TEXT ·CounterfactualFloat32AVX2Asm(SB), NOSPLIT, $0-48
	MOVQ out+0(FP), DI
	MOVQ observedY+8(FP), SI
	MOVQ observedX+16(FP), R9
	MOVQ counterfactualX+24(FP), R10
	MOVSS slope+32(FP), X15
	MOVQ count+40(FP), CX
	VSHUFPS $0, X15, X15, X15
	VBROADCASTSS X15, Y15

cf_avx2_w8:
	CMPQ CX, $8
	JL   cf_avx2_w4

	VMOVUPS (R10), Y0
	VMOVUPS (R9), Y1
	VSUBPS  Y1, Y0, Y0
	VMULPS  Y15, Y0, Y0
	VMOVUPS (SI), Y2
	VADDPS  Y0, Y2, Y0
	VMOVUPS Y0, (DI)

	ADDQ $32, DI
	ADDQ $32, SI
	ADDQ $32, R9
	ADDQ $32, R10
	SUBQ $8, CX
	JMP  cf_avx2_w8

cf_avx2_w4:
	CMPQ CX, $4
	JL   cf_avx2_tail

	VMOVUPS (R10), X0
	VMOVUPS (R9), X1
	VSUBPS  X1, X0, X0
	VMULPS  X15, X0, X0
	VMOVUPS (SI), X2
	VADDPS  X0, X2, X0
	VMOVUPS X0, (DI)

	ADDQ $16, DI
	ADDQ $16, SI
	ADDQ $16, R9
	ADDQ $16, R10
	SUBQ $4, CX
	JMP  cf_avx2_w4

cf_avx2_tail:
	TESTQ CX, CX
	JZ   cf_avx2_done

cf_avx2_scalar:
	MOVSS (R10), X0
	SUBSS (R9), X0
	MULSS X15, X0
	ADDSS (SI), X0
	MOVSS X0, (DI)
	ADDQ  $4, DI
	ADDQ  $4, SI
	ADDQ  $4, R9
	ADDQ  $4, R10
	DECQ  CX
	JNZ   cf_avx2_scalar

cf_avx2_done:
	RET

// func StridedDotFloat32AVX2Asm(values *float32, stride int, weights *float32, count int) float32
TEXT ·StridedDotFloat32AVX2Asm(SB), NOSPLIT, $0-32
	MOVQ values+0(FP), SI
	MOVQ stride+8(FP), R9
	MOVQ weights+16(FP), DI
	MOVQ count+24(FP), CX

	TESTQ CX, CX
	JZ   strided_avx2_zero

	MOVQ R9, R10
	SHLQ $2, R10

	VXORPD Y0, Y0, Y0
	VPBROADCASTD R9, Y5

strided_avx2_w8:
	CMPQ CX, $8
	JL   strided_avx2_w4

	VMOVDQA ·stridedDotIota8AVX2<>(SB), Y6
	VPMULLD Y5, Y6, Y6
	VPSLLD  $2, Y6, Y6
	VXORPS  Y13, Y13, Y13
	VGATHERDPS Y13, (SI)(Y6*1), Y7
	VMOVUPS (DI), Y8

	VEXTRACTF128 $0, Y7, X1
	VEXTRACTF128 $0, Y8, X2
	VCVTPS2PD X1, Y10
	VCVTPS2PD X2, Y11
	VFMADD231PD Y0, Y11, Y10

	VEXTRACTF128 $1, Y7, X1
	VEXTRACTF128 $1, Y8, X2
	VCVTPS2PD X1, Y10
	VCVTPS2PD X2, Y11
	VFMADD231PD Y0, Y11, Y10

	ADDQ $32, DI
	MOVQ R10, AX
	SHLQ $3, AX
	ADDQ AX, SI
	SUBQ $8, CX
	JMP  strided_avx2_w8

strided_avx2_w4:
	CMPQ CX, $4
	JL   strided_avx2_tail

	VMOVDQA ·stridedDotIota4AVX2<>(SB), Y6
	VPMULLD Y5, Y6, Y6
	VPSLLD  $2, Y6, Y6
	VXORPS  X13, X13, X13
	VGATHERDPS X13, (SI)(X6*1), X7
	VMOVUPS (DI), X8

	VCVTPS2PD X7, Y10
	VCVTPS2PD X8, Y11
	VFMADD231PD Y0, Y11, Y10

	ADDQ $16, DI
	MOVQ R10, AX
	SHLQ $2, AX
	ADDQ AX, SI
	SUBQ $4, CX
	JMP  strided_avx2_w4

strided_avx2_tail:
	TESTQ CX, CX
	JZ   strided_avx2_reduce

strided_avx2_scalar:
	MOVSS (SI), X1
	VCVTPS2PD X1, Y10
	MOVSS (DI), X2
	VCVTPS2PD X2, Y11
	VFMADD231PD Y0, Y11, Y10

	ADDQ R10, SI
	ADDQ $4, DI
	DECQ CX
	JNZ  strided_avx2_scalar

strided_avx2_reduce:
	VHADDPD Y1, Y0, Y0
	VEXTRACTF128 $0, Y0, X0
	CVTSD2SS X0, X0
	MOVSS X0, ret+32(FP)
	RET

strided_avx2_zero:
	XORPS X0, X0
	MOVSS X0, ret+32(FP)
	RET

DATA ·stridedDotIota4AVX2<>+0(SB)/4, $0
DATA ·stridedDotIota4AVX2<>+4(SB)/4, $1
DATA ·stridedDotIota4AVX2<>+8(SB)/4, $2
DATA ·stridedDotIota4AVX2<>+12(SB)/4, $3
GLOBL ·stridedDotIota4AVX2<>(SB), RODATA, $16

DATA ·stridedDotIota8AVX2<>+0(SB)/4, $0
DATA ·stridedDotIota8AVX2<>+4(SB)/4, $1
DATA ·stridedDotIota8AVX2<>+8(SB)/4, $2
DATA ·stridedDotIota8AVX2<>+12(SB)/4, $3
DATA ·stridedDotIota8AVX2<>+16(SB)/4, $4
DATA ·stridedDotIota8AVX2<>+20(SB)/4, $5
DATA ·stridedDotIota8AVX2<>+24(SB)/4, $6
DATA ·stridedDotIota8AVX2<>+28(SB)/4, $7
GLOBL ·stridedDotIota8AVX2<>(SB), RODATA, $32
