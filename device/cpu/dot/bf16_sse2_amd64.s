#include "textflag.h"
#include "../sse2_bf16_macros.inc"

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

	BF16_DOT_W4_SSE2(SI, DI)

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
	PACK_BF16_SCALAR_F32_X0_TO(ret+24(FP))
	RET

dot_bf16_sse2_zero:
	XORPS X0, X0
	PACK_BF16_SCALAR_F32_X0_TO(ret+24(FP))
	RET
