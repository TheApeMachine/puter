#include "textflag.h"

DATA redAbsMaskSSE2<>+0(SB)/4, $0x7fffffff
DATA redAbsMaskSSE2<>+4(SB)/4, $0x7fffffff
DATA redAbsMaskSSE2<>+8(SB)/4, $0x7fffffff
DATA redAbsMaskSSE2<>+12(SB)/4, $0x7fffffff
GLOBL redAbsMaskSSE2<>(SB), RODATA|NOPTR, $16

DATA redOneF32SSE2<>+0(SB)/4, $0x3f800000
GLOBL redOneF32SSE2<>(SB), RODATA|NOPTR, $4

// func SumFloat32SSE2Asm(src *float32, count int) float32
TEXT ·SumFloat32SSE2Asm(SB), NOSPLIT, $0-20
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX
	TESTQ CX, CX
	JZ   sum_sse2_zero

	XORPD X0, X0

sum_sse2_w4:
	CMPQ CX, $4
	JL   sum_sse2_tail

	VMOVUPS (SI), X2
	VCVTPS2PD X2, X3
	ADDPD   X3, X0

	ADDQ $16, SI
	SUBQ $4, CX
	JMP  sum_sse2_w4

sum_sse2_tail:
	TESTQ CX, CX
	JZ   sum_sse2_reduce

sum_sse2_scalar:
	MOVSS (SI), X2
	CVTSS2SD X2, X2
	ADDSD  X2, X0
	ADDQ  $4, SI
	DECQ  CX
	JNZ  sum_sse2_scalar

sum_sse2_reduce:
	ADDSD X0, X0
	MOVSD X0, X1
	UNPCKHPD X0, X1
	ADDSD X1, X0
	CVTSD2SS X0, X0
	MOVSS X0, ret+16(FP)
	RET

sum_sse2_zero:
	XORPS X0, X0
	MOVSS X0, ret+16(FP)
	RET

// func ProdFloat32SSE2Asm(src *float32, count int) float32
TEXT ·ProdFloat32SSE2Asm(SB), NOSPLIT, $0-20
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX
	TESTQ CX, CX
	JZ   prod_sse2_zero

	MOVSS redOneF32SSE2<>(SB), X0

prod_sse2_w4:
	CMPQ CX, $4
	JL   prod_sse2_tail

	VMOVUPS (SI), X1
	MULPS X1, X0

	ADDQ $16, SI
	SUBQ $4, CX
	JMP  prod_sse2_w4

prod_sse2_tail:
	TESTQ CX, CX
	JZ   prod_sse2_fold

prod_sse2_scalar:
	MOVSS (SI), X1
	MULSS X1, X0
	ADDQ  $4, SI
	DECQ  CX
	JNZ  prod_sse2_scalar

prod_sse2_fold:
	MOVAPS X0, X1
	SHUFPS $2, X0, X1
	MULPS  X1, X0
	MOVAPS X0, X1
	SHUFPS $1, X0, X1
	MULPS  X1, X0
	MOVSS X0, ret+16(FP)
	RET

prod_sse2_zero:
	MOVSS redOneF32SSE2<>(SB), X0
	MOVSS X0, ret+16(FP)
	RET

// func ReduceMaxFloat32SSE2Asm(src *float32, count int) float32
TEXT ·ReduceMaxFloat32SSE2Asm(SB), NOSPLIT, $0-20
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX
	TESTQ CX, CX
	JZ   max_sse2_zero

	MOVSS (SI), X0
	ADDQ $4, SI
	DECQ CX

max_sse2_w4:
	CMPQ CX, $4
	JL   max_sse2_tail

	VMOVUPS (SI), X1
	MAXPS X1, X0

	ADDQ $16, SI
	SUBQ $4, CX
	JMP  max_sse2_w4

max_sse2_tail:
	TESTQ CX, CX
	JZ   max_sse2_fold

max_sse2_scalar:
	MOVSS (SI), X1
	MAXSS X1, X0
	ADDQ  $4, SI
	DECQ  CX
	JNZ  max_sse2_scalar

max_sse2_fold:
	MOVAPS X0, X1
	SHUFPS $2, X0, X1
	MAXPS  X1, X0
	MOVAPS X0, X1
	SHUFPS $1, X0, X1
	MAXPS  X1, X0
	MOVSS X0, ret+16(FP)
	RET

max_sse2_zero:
	XORPS X0, X0
	MOVSS X0, ret+16(FP)
	RET

// func ReduceMinFloat32SSE2Asm(src *float32, count int) float32
TEXT ·ReduceMinFloat32SSE2Asm(SB), NOSPLIT, $0-20
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX
	TESTQ CX, CX
	JZ   min_sse2_zero

	MOVSS (SI), X0
	ADDQ $4, SI
	DECQ CX

min_sse2_w4:
	CMPQ CX, $4
	JL   min_sse2_tail

	VMOVUPS (SI), X1
	MINPS X1, X0

	ADDQ $16, SI
	SUBQ $4, CX
	JMP  min_sse2_w4

min_sse2_tail:
	TESTQ CX, CX
	JZ   min_sse2_fold

min_sse2_scalar:
	MOVSS (SI), X1
	MINSS X1, X0
	ADDQ  $4, SI
	DECQ  CX
	JNZ  min_sse2_scalar

min_sse2_fold:
	MOVAPS X0, X1
	SHUFPS $2, X0, X1
	MINPS  X1, X0
	MOVAPS X0, X1
	SHUFPS $1, X0, X1
	MINPS  X1, X0
	MOVSS X0, ret+16(FP)
	RET

min_sse2_zero:
	XORPS X0, X0
	MOVSS X0, ret+16(FP)
	RET

// func L1NormFloat32SSE2Asm(src *float32, count int) float32
TEXT ·L1NormFloat32SSE2Asm(SB), NOSPLIT, $0-20
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX
	TESTQ CX, CX
	JZ   l1_sse2_zero

	XORPD X0, X0
	MOVUPS redAbsMaskSSE2<>(SB), X6

l1_sse2_w4:
	CMPQ CX, $4
	JL   l1_sse2_tail

	VMOVUPS (SI), X2
	ANDPS X6, X2
	VCVTPS2PD X2, X3
	ADDPD   X3, X0

	ADDQ $16, SI
	SUBQ $4, CX
	JMP  l1_sse2_w4

l1_sse2_tail:
	TESTQ CX, CX
	JZ   l1_sse2_reduce

l1_sse2_scalar:
	MOVSS (SI), X2
	ANDPS X6, X2
	CVTSS2SD X2, X2
	ADDSD  X2, X0
	ADDQ  $4, SI
	DECQ  CX
	JNZ  l1_sse2_scalar

l1_sse2_reduce:
	ADDSD X0, X0
	MOVSD X0, X1
	UNPCKHPD X0, X1
	ADDSD X1, X0
	CVTSD2SS X0, X0
	MOVSS X0, ret+16(FP)
	RET

l1_sse2_zero:
	XORPS X0, X0
	MOVSS X0, ret+16(FP)
	RET
