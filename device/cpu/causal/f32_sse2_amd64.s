// SPDX-License-Identifier: Apache-2.0
// SSE2 float32 causal kernels: CATE subtract, counterfactual, strided dot.
#include "textflag.h"

// func CateFloat32SSE2Asm(treated, control, out *float32, count int)
TEXT ·CateFloat32SSE2Asm(SB), NOSPLIT, $0-32
	MOVQ treated+0(FP), DI
	MOVQ control+8(FP), SI
	MOVQ out+16(FP), R8
	MOVQ count+24(FP), CX

cate_sse2_w4:
	CMPQ CX, $4
	JL   cate_sse2_tail

	MOVUPS (DI), X0
	MOVUPS (SI), X1
	SUBPS  X1, X0
	MOVUPS X0, (R8)

	ADDQ $16, DI
	ADDQ $16, SI
	ADDQ $16, R8
	SUBQ $4, CX
	JMP  cate_sse2_w4

cate_sse2_tail:
	TESTQ CX, CX
	JZ   cate_sse2_done

cate_sse2_scalar:
	MOVSS (DI), X0
	SUBSS (SI), X0
	MOVSS X0, (R8)
	ADDQ  $4, DI
	ADDQ  $4, SI
	ADDQ  $4, R8
	DECQ  CX
	JNZ   cate_sse2_scalar

cate_sse2_done:
	RET

// func CounterfactualFloat32SSE2Asm(out, observedY, observedX, counterfactualX *float32, slope float32, count int)
TEXT ·CounterfactualFloat32SSE2Asm(SB), NOSPLIT, $0-48
	MOVQ out+0(FP), DI
	MOVQ observedY+8(FP), SI
	MOVQ observedX+16(FP), R9
	MOVQ counterfactualX+24(FP), R10
	MOVSS slope+32(FP), X15
	MOVQ count+40(FP), CX
	SHUFPS $0, X15, X15

cf_sse2_w4:
	CMPQ CX, $4
	JL   cf_sse2_tail

	MOVUPS (R10), X0
	MOVUPS (R9), X1
	SUBPS  X1, X0
	MULPS  X15, X0
	MOVUPS (SI), X2
	ADDPS  X0, X2
	MOVUPS X2, (DI)

	ADDQ $16, DI
	ADDQ $16, SI
	ADDQ $16, R9
	ADDQ $16, R10
	SUBQ $4, CX
	JMP  cf_sse2_w4

cf_sse2_tail:
	TESTQ CX, CX
	JZ   cf_sse2_done

cf_sse2_scalar:
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
	JNZ   cf_sse2_scalar

cf_sse2_done:
	RET

// func StridedDotFloat32SSE2Asm(values *float32, stride int, weights *float32, count int) float32
TEXT ·StridedDotFloat32SSE2Asm(SB), NOSPLIT, $0-32
	MOVQ values+0(FP), SI
	MOVQ stride+8(FP), R9
	MOVQ weights+16(FP), DI
	MOVQ count+24(FP), CX

	TESTQ CX, CX
	JZ   strided_sse2_zero

	MOVQ R9, R10
	SHLQ $2, R10

	XORPD X0, X0

strided_sse2_w4:
	CMPQ CX, $4
	JL   strided_sse2_tail

	MOVUPS (DI), X8

	XORPS X7, X7
	MOVSS (SI), X7
	MOVQ  SI, R11
	ADDQ  R10, R11
	MOVSS (R11), X1
	UNPCKLPS X1, X7

	MOVQ  SI, R11
	ADDQ  R10, R11
	ADDQ  R10, R11
	MOVSS (R11), X2
	MOVQ  SI, R12
	MOVQ  R10, R13
	SHLQ  $1, R13
	ADDQ  R13, R12
	ADDQ  R10, R12
	MOVSS (R12), X3
	UNPCKLPS X3, X2
	MOVHLPS X2, X7

	MULPS X8, X7

	CVTPS2PD X7, X4
	SHUFPS $0xEE, X7, X5
	CVTPS2PD X5, X6
	ADDPD  X6, X4
	ADDPD  X4, X0

	ADDQ $16, DI
	MOVQ R10, AX
	SHLQ $2, AX
	ADDQ AX, SI
	SUBQ $4, CX
	JMP  strided_sse2_w4

strided_sse2_tail:
	TESTQ CX, CX
	JZ   strided_sse2_reduce

strided_sse2_scalar:
	MOVSS (SI), X1
	CVTPS2PD X1, X4
	MOVSS (DI), X2
	CVTPS2PD X2, X5
	MULPD  X5, X4
	ADDPD  X4, X0

	ADDQ R10, SI
	ADDQ $4, DI
	DECQ CX
	JNZ  strided_sse2_scalar

strided_sse2_reduce:
	MOVAPD X0, X1
	SHUFPD $1, X0, X1
	ADDPD  X1, X0
	CVTSD2SS X0, X0
	MOVSS X0, ret+32(FP)
	RET

strided_sse2_zero:
	XORPS X0, X0
	MOVSS X0, ret+32(FP)
	RET
