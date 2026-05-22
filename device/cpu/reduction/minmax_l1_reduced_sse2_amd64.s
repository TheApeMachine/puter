#include "textflag.h"

DATA l1ReducedAbsMaskSSE2<>+0(SB)/4, $0x7fffffff
DATA l1ReducedAbsMaskSSE2<>+4(SB)/4, $0x7fffffff
DATA l1ReducedAbsMaskSSE2<>+8(SB)/4, $0x7fffffff
DATA l1ReducedAbsMaskSSE2<>+12(SB)/4, $0x7fffffff
GLOBL l1ReducedAbsMaskSSE2<>(SB), RODATA|NOPTR, $16

#define BF16_MINMAX_FOLD_SSE2 \
	MOVAPS X0, X1; \
	SHUFPS $0xEE, X0, X1; \
	VMINPS X1, X0, X0; \
	MOVAPS X0, X1; \
	SHUFPS $0x55, X0, X1; \
	VMINPS X1, X0, X0

#define BF16_MAX_FOLD_SSE2 \
	MOVAPS X0, X1; \
	SHUFPS $0xEE, X0, X1; \
	VMAXPS X1, X0, X0; \
	MOVAPS X0, X1; \
	SHUFPS $0x55, X0, X1; \
	VMAXPS X1, X0, X0

#define BF16_L1_HSUM_XMM4_INTO_X0 \
	MOVAPS X4, X1; \
	SHUFPS $0xEE, X4, X1; \
	VADDPS X1, X4, X4; \
	MOVAPS X4, X1; \
	SHUFPS $0x55, X4, X1; \
	VADDPS X1, X4, X4; \
	ADDSS X4, X0

// func MinBFloat16SSE2Asm(src *uint16, count int) float32
TEXT ·MinBFloat16SSE2Asm(SB), NOSPLIT, $0-20
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX

	TESTQ CX, CX
	JZ    min_bf16_sse2_zero

	MOVWLZX (SI), AX
	SHLQ  $16, AX
	VMOVD X0, AX
	MOVAPS X0, X1
	SHUFPS $0, X0, X1
	MOVAPS X1, X0

	ADDQ $2, SI
	DECQ CX

min_bf16_sse2_w4:
	CMPQ CX, $4
	JL    min_bf16_sse2_tail

	VMOVDQU X1, (SI)
	VPXOR   X3, X3, X3
	VPUNPCKLWD X3, X1, X4
	VPUNPCKHWD X3, X1, X5
	VPSLLD  $16, X4, X4
	VPSLLD  $16, X5, X5
	VMINPS  X4, X0, X0
	VMINPS  X5, X0, X0

	ADDQ $8, SI
	SUBQ $4, CX
	JMP  min_bf16_sse2_w4

min_bf16_sse2_tail:
	BF16_MINMAX_FOLD_SSE2

	TESTQ CX, CX
	JZ    min_bf16_sse2_store

min_bf16_sse2_scalar:
	MOVWLZX (SI), AX
	SHLQ  $16, AX
	VMOVD X1, AX
	MINSS X1, X0

	ADDQ $2, SI
	DECQ CX
	JNZ  min_bf16_sse2_scalar

min_bf16_sse2_store:
	MOVSS X0, ret+16(FP)
	RET

min_bf16_sse2_zero:
	XORPS X0, X0
	MOVSS X0, ret+16(FP)
	RET

// func MaxBFloat16SSE2Asm(src *uint16, count int) float32
TEXT ·MaxBFloat16SSE2Asm(SB), NOSPLIT, $0-20
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX

	TESTQ CX, CX
	JZ    max_bf16_sse2_zero

	MOVWLZX (SI), AX
	SHLQ  $16, AX
	VMOVD X0, AX
	MOVAPS X0, X1
	SHUFPS $0, X0, X1
	MOVAPS X1, X0

	ADDQ $2, SI
	DECQ CX

max_bf16_sse2_w4:
	CMPQ CX, $4
	JL    max_bf16_sse2_tail

	VMOVDQU X1, (SI)
	VPXOR   X3, X3, X3
	VPUNPCKLWD X3, X1, X4
	VPUNPCKHWD X3, X1, X5
	VPSLLD  $16, X4, X4
	VPSLLD  $16, X5, X5
	VMAXPS  X4, X0, X0
	VMAXPS  X5, X0, X0

	ADDQ $8, SI
	SUBQ $4, CX
	JMP  max_bf16_sse2_w4

max_bf16_sse2_tail:
	BF16_MAX_FOLD_SSE2

	TESTQ CX, CX
	JZ    max_bf16_sse2_store

max_bf16_sse2_scalar:
	MOVWLZX (SI), AX
	SHLQ  $16, AX
	VMOVD X1, AX
	MAXSS X1, X0

	ADDQ $2, SI
	DECQ CX
	JNZ  max_bf16_sse2_scalar

max_bf16_sse2_store:
	MOVSS X0, ret+16(FP)
	RET

max_bf16_sse2_zero:
	XORPS X0, X0
	MOVSS X0, ret+16(FP)
	RET

// func L1NormBFloat16SSE2Asm(src *uint16, count int) float32
TEXT ·L1NormBFloat16SSE2Asm(SB), NOSPLIT, $0-20
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX

	TESTQ CX, CX
	JZ    l1_bf16_sse2_zero

	CMPQ CX, $1
	JNE   l1_bf16_sse2_multi

	MOVWLZX (SI), AX
	SHLQ  $16, AX
	VMOVD X0, AX
	MOVUPS l1ReducedAbsMaskSSE2<>(SB), X6
	ANDPS X6, X0
	MOVSS X0, ret+16(FP)
	RET

l1_bf16_sse2_multi:
	XORPS X0, X0
	MOVUPS l1ReducedAbsMaskSSE2<>(SB), X6

l1_bf16_sse2_w4:
	CMPQ CX, $4
	JL    l1_bf16_sse2_tail

	VMOVDQU X1, (SI)
	VPXOR   X3, X3, X3
	VPUNPCKLWD X3, X1, X4
	VPUNPCKHWD X3, X1, X5
	VPSLLD  $16, X4, X4
	VPSLLD  $16, X5, X5
	VINSERTF128 $1, X5, Y4, Y4
	VEXTRACTF128 $0, Y4, X4
	VANDPS  X6, X4, X4
	BF16_L1_HSUM_XMM4_INTO_X0

	ADDQ $8, SI
	SUBQ $4, CX
	JMP  l1_bf16_sse2_w4

l1_bf16_sse2_tail:
	TESTQ CX, CX
	JZ    l1_bf16_sse2_store

l1_bf16_sse2_scalar:
	MOVWLZX (SI), AX
	SHLQ  $16, AX
	VMOVD X1, AX
	VANDPS X6, X1, X1
	ADDSS X1, X0

	ADDQ $2, SI
	DECQ CX
	JNZ  l1_bf16_sse2_scalar

l1_bf16_sse2_store:
	MOVSS X0, ret+16(FP)
	RET

l1_bf16_sse2_zero:
	XORPS X0, X0
	MOVSS X0, ret+16(FP)
	RET

// func MinFloat16SSE2Asm(src *uint16, count int) float32
TEXT ·MinFloat16SSE2Asm(SB), NOSPLIT, $0-20
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX

	TESTQ CX, CX
	JZ    min_fp16_sse2_zero

	MOVWLZX (SI), AX
	VMOVD X0, AX
	VCVTPH2PS X0, X0
	MOVAPS X0, X1
	SHUFPS $0, X0, X1
	MOVAPS X1, X0

	ADDQ $2, SI
	DECQ CX

min_fp16_sse2_w4:
	CMPQ CX, $4
	JL    min_fp16_sse2_tail

	VMOVDQU X1, (SI)
	VCVTPH2PS X1, X4
	VMINPS  X4, X0, X0

	ADDQ $8, SI
	SUBQ $4, CX
	JMP  min_fp16_sse2_w4

min_fp16_sse2_tail:
	BF16_MINMAX_FOLD_SSE2

	TESTQ CX, CX
	JZ    min_fp16_sse2_store

min_fp16_sse2_scalar:
	MOVWLZX (SI), AX
	VMOVD X1, AX
	VCVTPH2PS X1, X1
	MINSS X1, X0

	ADDQ $2, SI
	DECQ CX
	JNZ  min_fp16_sse2_scalar

min_fp16_sse2_store:
	MOVSS X0, ret+16(FP)
	RET

min_fp16_sse2_zero:
	XORPS X0, X0
	MOVSS X0, ret+16(FP)
	RET

// func MaxFloat16SSE2Asm(src *uint16, count int) float32
TEXT ·MaxFloat16SSE2Asm(SB), NOSPLIT, $0-20
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX

	TESTQ CX, CX
	JZ    max_fp16_sse2_zero

	MOVWLZX (SI), AX
	VMOVD X0, AX
	VCVTPH2PS X0, X0
	MOVAPS X0, X1
	SHUFPS $0, X0, X1
	MOVAPS X1, X0

	ADDQ $2, SI
	DECQ CX

max_fp16_sse2_w4:
	CMPQ CX, $4
	JL    max_fp16_sse2_tail

	VMOVDQU X1, (SI)
	VCVTPH2PS X1, X4
	VMAXPS  X4, X0, X0

	ADDQ $8, SI
	SUBQ $4, CX
	JMP  max_fp16_sse2_w4

max_fp16_sse2_tail:
	BF16_MAX_FOLD_SSE2

	TESTQ CX, CX
	JZ    max_fp16_sse2_store

max_fp16_sse2_scalar:
	MOVWLZX (SI), AX
	VMOVD X1, AX
	VCVTPH2PS X1, X1
	MAXSS X1, X0

	ADDQ $2, SI
	DECQ CX
	JNZ  max_fp16_sse2_scalar

max_fp16_sse2_store:
	MOVSS X0, ret+16(FP)
	RET

max_fp16_sse2_zero:
	XORPS X0, X0
	MOVSS X0, ret+16(FP)
	RET

// func L1NormFloat16SSE2Asm(src *uint16, count int) float32
TEXT ·L1NormFloat16SSE2Asm(SB), NOSPLIT, $0-20
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX

	TESTQ CX, CX
	JZ    l1_fp16_sse2_zero

	CMPQ CX, $1
	JNE   l1_fp16_sse2_multi

	MOVWLZX (SI), AX
	VMOVD X0, AX
	VCVTPH2PS X0, X0
	MOVUPS l1ReducedAbsMaskSSE2<>(SB), X6
	ANDPS X6, X0
	MOVSS X0, ret+16(FP)
	RET

l1_fp16_sse2_multi:
	XORPS X0, X0
	MOVUPS l1ReducedAbsMaskSSE2<>(SB), X6

l1_fp16_sse2_w4:
	CMPQ CX, $4
	JL    l1_fp16_sse2_tail

	VMOVDQU X1, (SI)
	VCVTPH2PS X1, X4
	VANDPS  X6, X4, X4
	BF16_L1_HSUM_XMM4_INTO_X0

	ADDQ $8, SI
	SUBQ $4, CX
	JMP  l1_fp16_sse2_w4

l1_fp16_sse2_tail:
	TESTQ CX, CX
	JZ    l1_fp16_sse2_store

l1_fp16_sse2_scalar:
	MOVWLZX (SI), AX
	VMOVD X1, AX
	VCVTPH2PS X1, X1
	VANDPS X6, X1, X1
	VADDSS X1, X0, X0

	ADDQ $2, SI
	DECQ CX
	JNZ  l1_fp16_sse2_scalar

l1_fp16_sse2_store:
	MOVSS X0, ret+16(FP)
	RET

l1_fp16_sse2_zero:
	XORPS X0, X0
	MOVSS X0, ret+16(FP)
	RET
