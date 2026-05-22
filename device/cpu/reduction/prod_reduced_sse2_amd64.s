#include "textflag.h"

DATA prodOneF32SSE2<>+0(SB)/4, $0x3f800000
GLOBL prodOneF32SSE2<>(SB), RODATA|NOPTR, $4

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

	VMOVDQU X1, (SI)
	VPXOR   X3, X3, X3
	VPUNPCKLWD X3, X1, X4
	VPUNPCKHWD X3, X1, X5
	VPSLLD  $16, X4, X4
	VPSLLD  $16, X5, X5
	VMULPS  X4, X0, X0
	VMULPS  X5, X0, X0

	ADDQ $8, SI
	SUBQ $4, CX
	JMP  prod_bf16_sse2_w4

prod_bf16_sse2_tail:
	MOVAPS X0, X1
	SHUFPS $0xEE, X0, X1
	VMULPS X1, X0, X0
	MOVAPS X0, X1
	SHUFPS $0x55, X0, X1
	VMULPS X1, X0, X0

	TESTQ CX, CX
	JZ    prod_bf16_sse2_store

prod_bf16_sse2_scalar:
	MOVWLZX (SI), AX
	SHLQ  $16, AX
	VMOVD X1, AX
	VMULSS X1, X0, X0

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

	VMOVDQU X1, (SI)
	VCVTPH2PS X1, X4
	VMULPS  X4, X0, X0

	ADDQ $8, SI
	SUBQ $4, CX
	JMP  prod_fp16_sse2_w4

prod_fp16_sse2_tail:
	MOVAPS X0, X1
	SHUFPS $0xEE, X0, X1
	VMULPS X1, X0, X0
	MOVAPS X0, X1
	SHUFPS $0x55, X0, X1
	VMULPS X1, X0, X0

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
