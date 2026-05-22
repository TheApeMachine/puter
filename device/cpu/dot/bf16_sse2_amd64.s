#include "textflag.h"

#define VCVTPS2PH_X0_X2 WORD $0xC4E3; WORD $0x7D1D; BYTE $0xD0; BYTE $0x00

#define NARROW_BF16_F32_X0_TO_RET \
	MOVL  X0, AX; \
	SHRQ  $16, AX; \
	MOVW  AX, ret+24(FP)

// func DotBFloat16SSE2Asm(left, right *uint16, count int) uint16
TEXT ·DotBFloat16SSE2Asm(SB), NOSPLIT, $0-26
	MOVQ left+0(FP), SI
	MOVQ right+8(FP), DI
	MOVQ count+16(FP), CX

	TESTQ CX, CX
	JZ    dot_bf16_sse2_zero

	XORPS X0, X0

dot_bf16_sse2_w4:
	CMPQ CX, $4
	JL   dot_bf16_sse2_reduce

	VMOVDQU X1, (SI)
	VMOVDQU X2, (DI)
	VPXOR   X3, X3, X3
	VPUNPCKLWD X3, X1, X4
	VPUNPCKLWD X3, X2, X5
	VPSLLD  $16, X4, X4
	VPSLLD  $16, X5, X5
	MULPS   X4, X5
	ADDPS   X5, X0

	VPUNPCKHWD X3, X1, X4
	VPUNPCKHWD X3, X2, X5
	VPSLLD  $16, X4, X4
	VPSLLD  $16, X5, X5
	MULPS   X4, X5
	ADDPS   X5, X0

	ADDQ $8, SI
	ADDQ $8, DI
	SUBQ $4, CX
	JMP  dot_bf16_sse2_w4

dot_bf16_sse2_reduce:
	MOVAPS X0, X1
	SHUFPS $0xEE, X0, X1
	ADDPS  X1, X0
	MOVAPS X0, X1
	SHUFPS $0x55, X0, X1
	ADDPS  X1, X0

	TESTQ CX, CX
	JZ    dot_bf16_sse2_store

dot_bf16_sse2_scalar:
	MOVWLZX (SI), AX
	SHLQ  $16, AX
	VMOVD X1, AX
	MOVWLZX (DI), DX
	SHLQ  $16, DX
	VMOVD X2, DX
	VMULSS X2, X1, X1
	VADDSS X1, X0, X0

	ADDQ $2, SI
	ADDQ $2, DI
	DECQ CX
	JNZ  dot_bf16_sse2_scalar

dot_bf16_sse2_store:
	NARROW_BF16_F32_X0_TO_RET
	RET

dot_bf16_sse2_zero:
	XORPS X0, X0
	NARROW_BF16_F32_X0_TO_RET
	RET
