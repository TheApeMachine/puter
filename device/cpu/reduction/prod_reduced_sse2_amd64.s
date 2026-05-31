#include "textflag.h"
#include "../sse2_bf16_macros.inc"
#include "../f16c_fp16_macros.inc"

DATA prodOneF32SSE2<>+0(SB)/4, $0x3f800000
GLOBL prodOneF32SSE2<>(SB), RODATA|NOPTR, $4

#define BF16_PROD_HMUL_XMM4_INTO_X0 \
	MOVAPS X4, X1; \
	SHUFPS $0xEE, X4, X1; \
	VMULPS X1, X4, X4; \
	MOVAPS X4, X1; \
	SHUFPS $0x55, X4, X1; \
	VMULPS X1, X4, X4; \
	MULSS X4, X0

// func ProdBFloat16SSE2Asm(src *uint16, count int) float32
TEXT ·ProdBFloat16SSE2Asm(SB), NOSPLIT, $0-20
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX

	TESTQ CX, CX
	JZ    prod_bf16_sse2_zero

	MOVSS prodOneF32SSE2<>(SB), X0

prod_bf16_sse2_w4:
	CMPQ CX, $4
	JL    prod_bf16_sse2_tail

	BF16_WIDEN_SSE2_4(SI, X4, X5)
	BF16_PROD_HMUL_XMM4_INTO_X0

	ADDQ $8, SI
	SUBQ $4, CX
	JMP  prod_bf16_sse2_w4

prod_bf16_sse2_tail:
	TESTQ CX, CX
	JZ    prod_bf16_sse2_store

prod_bf16_sse2_scalar:
	MOVWLZX (SI), AX
	SHLQ  $16, AX
	VMOVD X1, AX
	MULSS X1, X0

	ADDQ $2, SI
	DECQ CX
	JNZ  prod_bf16_sse2_scalar

prod_bf16_sse2_store:
	MOVSS X0, ret+16(FP)
	RET

prod_bf16_sse2_zero:
	XORPS X0, X0
	MOVSS X0, ret+16(FP)
	RET

// func ProdFloat16SSE2Asm(src *uint16, count int) float32
TEXT ·ProdFloat16SSE2Asm(SB), NOSPLIT, $0-20
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX

	TESTQ CX, CX
	JZ    prod_fp16_sse2_zero

	MOVSS prodOneF32SSE2<>(SB), X0

prod_fp16_sse2_w4:
	CMPQ CX, $4
	JL    prod_fp16_sse2_tail

	FP16_WIDEN_SSE2_4(SI, X4)
	BF16_PROD_HMUL_XMM4_INTO_X0

	ADDQ $8, SI
	SUBQ $4, CX
	JMP  prod_fp16_sse2_w4

prod_fp16_sse2_tail:
	TESTQ CX, CX
	JZ    prod_fp16_sse2_store

prod_fp16_sse2_scalar:
	MOVWLZX (SI), AX
	VMOVD X1, AX
	VCVTPH2PS X1, X1
	VMULSS X1, X0, X0

	ADDQ $2, SI
	DECQ CX
	JNZ  prod_fp16_sse2_scalar

prod_fp16_sse2_store:
	MOVSS X0, ret+16(FP)
	RET

prod_fp16_sse2_zero:
	XORPS X0, X0
	MOVSS X0, ret+16(FP)
	RET
