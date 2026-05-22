#include "textflag.h"

#define VCVTPS2PH_X0_X2 WORD $0xC4E3; WORD $0x7D1D; BYTE $0xD0; BYTE $0x00

#define NARROW_FP16_F32_X0_TO_RET \
	VCVTPS2PH_X0_X2; \
	MOVL  X2, AX; \
	MOVW  AX, ret+16(FP)

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

	VMOVDQU X1, (SI)
	VCVTPH2PS X1, X4
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
	NARROW_FP16_F32_X0_TO_RET
	RET

sum_fp16_sse2_zero:
	XORPS X0, X0
	NARROW_FP16_F32_X0_TO_RET
	RET
