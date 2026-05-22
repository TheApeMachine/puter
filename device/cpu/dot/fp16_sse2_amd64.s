#include "textflag.h"

#define VCVTPS2PH_X0_X2 WORD $0xC4E3; WORD $0x7D1D; BYTE $0xD0; BYTE $0x00

#define NARROW_FP16_F32_X0_TO_RET \
	VCVTPS2PH_X0_X2; \
	MOVL  X2, AX; \
	MOVW  AX, ret+24(FP)

// func DotFloat16SSE2Asm(left, right *uint16, count int) uint16
TEXT ·DotFloat16SSE2Asm(SB), NOSPLIT, $0-26
	MOVQ left+0(FP), SI
	MOVQ right+8(FP), DI
	MOVQ count+16(FP), CX

	TESTQ CX, CX
	JZ    dot_fp16_sse2_zero

	XORPS X0, X0

dot_fp16_sse2_w4:
	CMPQ CX, $4
	JL   dot_fp16_sse2_reduce

	VMOVDQU X1, (SI)
	VMOVDQU X2, (DI)
	VCVTPH2PS X1, X4
	VCVTPH2PS X2, X5
	MULPS   X4, X5
	ADDPS   X5, X0

	ADDQ $8, SI
	ADDQ $8, DI
	SUBQ $4, CX
	JMP  dot_fp16_sse2_w4

dot_fp16_sse2_reduce:
	MOVAPS X0, X1
	SHUFPS $0xEE, X0, X1
	ADDPS  X1, X0
	MOVAPS X0, X1
	SHUFPS $0x55, X0, X1
	ADDPS  X1, X0

	TESTQ CX, CX
	JZ    dot_fp16_sse2_store

dot_fp16_sse2_scalar:
	MOVWLZX (SI), AX
	VMOVD X1, AX
	VCVTPH2PS X1, X1
	MOVWLZX (DI), DX
	VMOVD X2, DX
	VCVTPH2PS X2, X2
	VMULSS X2, X1, X1
	VADDSS X1, X0, X0

	ADDQ $2, SI
	ADDQ $2, DI
	DECQ CX
	JNZ  dot_fp16_sse2_scalar

dot_fp16_sse2_store:
	NARROW_FP16_F32_X0_TO_RET
	RET

dot_fp16_sse2_zero:
	XORPS X0, X0
	NARROW_FP16_F32_X0_TO_RET
	RET
