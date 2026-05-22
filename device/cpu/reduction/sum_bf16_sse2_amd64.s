#include "textflag.h"

#define NARROW_BF16_F32_X0_TO_RET \
	MOVL  X0, AX; \
	SHRQ  $16, AX; \
	MOVW  AX, ret+16(FP)

// func SumBFloat16SSE2Asm(src *uint16, count int) uint16
TEXT ·SumBFloat16SSE2Asm(SB), NOSPLIT, $0-18
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX

	TESTQ CX, CX
	JZ    sum_bf16_sse2_zero

	XORPS X0, X0

sum_bf16_sse2_w4:
	CMPQ CX, $4
	JL   sum_bf16_sse2_reduce

	VMOVDQU X1, (SI)
	VPXOR   X3, X3, X3
	VPUNPCKLWD X3, X1, X4
	VPUNPCKHWD X3, X1, X5
	VPSLLD  $16, X4, X4
	VPSLLD  $16, X5, X5
	ADDPS   X4, X0
	ADDPS   X5, X0

	ADDQ $8, SI
	SUBQ $4, CX
	JMP  sum_bf16_sse2_w4

sum_bf16_sse2_reduce:
	MOVAPS X0, X1
	SHUFPS $0xEE, X0, X1
	ADDPS  X1, X0
	MOVAPS X0, X1
	SHUFPS $0x55, X0, X1
	ADDPS  X1, X0

	TESTQ CX, CX
	JZ    sum_bf16_sse2_store

sum_bf16_sse2_scalar:
	MOVWLZX (SI), AX
	SHLQ  $16, AX
	VMOVD X1, AX
	VADDSS X1, X0, X0

	ADDQ $2, SI
	DECQ CX
	JNZ  sum_bf16_sse2_scalar

sum_bf16_sse2_store:
	NARROW_BF16_F32_X0_TO_RET
	RET

sum_bf16_sse2_zero:
	XORPS X0, X0
	NARROW_BF16_F32_X0_TO_RET
	RET
