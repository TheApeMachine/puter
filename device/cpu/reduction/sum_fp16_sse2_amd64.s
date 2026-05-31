#include "textflag.h"
#include "../f16c_fp16_macros.inc"

// func SumFloat16SSE2Asm(src *uint16, count int) uint16
TEXT ·SumFloat16SSE2Asm(SB), NOSPLIT, $0-18
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX

	TESTQ CX, CX
	JZ    sum_fp16_sse2_zero

	XORPS X0, X0

sum_fp16_sse2_w4:
	CMPQ CX, $4
	JL   sum_fp16_sse2_reduce

	FP16_WIDEN_SSE2_4(SI, X4)
	ADDPS   X4, X0

	ADDQ $8, SI
	SUBQ $4, CX
	JMP  sum_fp16_sse2_w4

sum_fp16_sse2_reduce:
	MOVAPS X0, X1
	SHUFPS $0xEE, X0, X1
	ADDPS  X1, X0
	MOVAPS X0, X1
	SHUFPS $0x55, X0, X1
	ADDPS  X1, X0

	TESTQ CX, CX
	JZ    sum_fp16_sse2_store

sum_fp16_sse2_scalar:
	MOVWLZX (SI), AX
	VMOVD X1, AX
	VCVTPH2PS X1, X1
	VADDSS X1, X0, X0

	ADDQ $2, SI
	DECQ CX
	JNZ  sum_fp16_sse2_scalar

sum_fp16_sse2_store:
	FP16_NARROW_SCALAR_F32_X0_TO(ret+16(FP))
	RET

sum_fp16_sse2_zero:
	XORPS X0, X0
	FP16_NARROW_SCALAR_F32_X0_TO(ret+16(FP))
	RET
