#include "textflag.h"

#define WIDEN_BF16_4(src, xLo, xHi) \
	VMOVDQU X2, (src); \
	VPXOR  X3, X3, X3; \
	VPUNPCKLWD X3, X2, xLo; \
	VPUNPCKHWD X3, X2, xHi; \
	VPSLLD $16, xLo, xLo; \
	VPSLLD $16, xHi, xHi

#define NARROW_BF16_4(xLo, xHi, dst) \
	VPSRLD $16, xLo, xLo; \
	VPSRLD $16, xHi, xHi; \
	MOVL  xLo, AX; \
	MOVW  AX, (dst); \
	PEXTRD $1, xLo, AX; \
	MOVW  AX, 2(dst); \
	MOVL  xHi, AX; \
	MOVW  AX, 4(dst); \
	PEXTRD $1, xHi, AX; \
	MOVW  AX, 6(dst)

#define AI_BF16_ACCUM_F64_X(prodLo, prodHi, accumX) \
	CVTPS2PD prodLo, X8; \
	ADDPD X8, accumX; \
	MOVAPS prodLo, X9; \
	SHUFPS $0xEE, prodLo, X9; \
	CVTPS2PD X9, X8; \
	ADDPD X8, accumX; \
	CVTPS2PD prodHi, X8; \
	ADDPD X8, accumX; \
	MOVAPS prodHi, X9; \
	SHUFPS $0xEE, prodHi, X9; \
	CVTPS2PD X9, X8; \
	ADDPD X8, accumX

// func PrecisionWeightBFloat16SSE2Asm(errors, precision, output *uint16, count int)
TEXT ·PrecisionWeightBFloat16SSE2Asm(SB), NOSPLIT, $0-32
	MOVQ errors+0(FP), SI
	MOVQ precision+8(FP), DX
	MOVQ output+16(FP), DI
	MOVQ count+24(FP), CX

	TESTQ CX, CX
	JZ   ai_bf16_sse2_pw_done

ai_bf16_sse2_pw_w4:
	CMPQ CX, $4
	JL   ai_bf16_sse2_pw_tail

	WIDEN_BF16_4(SI, X4, X5)
	WIDEN_BF16_4(DX, X6, X7)
	MULPS X6, X4
	MULPS X7, X5
	NARROW_BF16_4(X4, X5, DI)

	ADDQ $8, SI
	ADDQ $8, DX
	ADDQ $8, DI
	SUBQ $4, CX
	JMP  ai_bf16_sse2_pw_w4

ai_bf16_sse2_pw_tail:
	TESTQ CX, CX
	JZ   ai_bf16_sse2_pw_done

ai_bf16_sse2_pw_scalar:
	MOVWLZX (SI), AX
	SHLQ  $16, AX
	VMOVD X2, AX
	MOVWLZX (DX), AX
	SHLQ  $16, AX
	VMOVD X3, AX
	MULSS X3, X2
	VPSRLD $16, X2, X2
	MOVL  X2, AX
	MOVW  AX, (DI)
	ADDQ $2, SI
	ADDQ $2, DX
	ADDQ $2, DI
	DECQ CX
	JNZ  ai_bf16_sse2_pw_scalar

ai_bf16_sse2_pw_done:
	RET

// func BeliefUpdateBFloat16SSE2Asm(likelihood, prior, output *uint16, count int)
TEXT ·BeliefUpdateBFloat16SSE2Asm(SB), NOSPLIT, $0-32
	MOVQ likelihood+0(FP), SI
	MOVQ prior+8(FP), DX
	MOVQ output+16(FP), DI
	MOVQ count+24(FP), CX

	TESTQ CX, CX
	JZ   ai_bf16_sse2_bu_done

	XORPD X0, X0

ai_bf16_sse2_bu_mul_w4:
	CMPQ CX, $4
	JL   ai_bf16_sse2_bu_mul_tail

	WIDEN_BF16_4(SI, X4, X5)
	WIDEN_BF16_4(DX, X6, X7)
	MULPS X6, X4
	MULPS X7, X5
	AI_BF16_ACCUM_F64_X(X4, X5, X0)
	NARROW_BF16_4(X4, X5, DI)

	ADDQ $8, SI
	ADDQ $8, DX
	ADDQ $8, DI
	SUBQ $4, CX
	JMP  ai_bf16_sse2_bu_mul_w4

ai_bf16_sse2_bu_mul_tail:
	TESTQ CX, CX
	JZ   ai_bf16_sse2_bu_reduce

ai_bf16_sse2_bu_mul_scalar:
	MOVWLZX (SI), AX
	SHLQ  $16, AX
	VMOVD X2, AX
	MOVWLZX (DX), AX
	SHLQ  $16, AX
	VMOVD X3, AX
	MULSS X3, X2
	MOVSS X2, X1
	CVTSS2SD X1, X1
	ADDSD X1, X0
	VPSRLD $16, X2, X2
	MOVL  X2, AX
	MOVW  AX, (DI)
	ADDQ $2, SI
	ADDQ $2, DX
	ADDQ $2, DI
	DECQ CX
	JNZ  ai_bf16_sse2_bu_mul_scalar

ai_bf16_sse2_bu_reduce:
	MOVAPS X0, X1
	SHUFPD $1, X1, X1
	ADDPD  X1, X0
	CVTSD2SS X0, X0

	XORPS X1, X1
	UCOMISS X0, X1
	JZ    ai_bf16_sse2_bu_done

	MOVSS aiOneBits<>(SB), X3
	DIVSS X2, X3
	SHUFPS $0, X3, X3

	MOVQ output+16(FP), DI
	MOVQ count+24(FP), CX

ai_bf16_sse2_bu_scale_w4:
	CMPQ CX, $4
	JL   ai_bf16_sse2_bu_scale_tail

	WIDEN_BF16_4(DI, X4, X5)
	MULPS X3, X4
	MULPS X3, X5
	NARROW_BF16_4(X4, X5, DI)

	ADDQ $8, DI
	SUBQ $4, CX
	JMP  ai_bf16_sse2_bu_scale_w4

ai_bf16_sse2_bu_scale_tail:
	TESTQ CX, CX
	JZ   ai_bf16_sse2_bu_done

ai_bf16_sse2_bu_scale_scalar:
	MOVWLZX (DI), AX
	SHLQ  $16, AX
	VMOVD X2, AX
	MULSS X3, X2
	VPSRLD $16, X2, X2
	MOVL  X2, AX
	MOVW  AX, (DI)
	ADDQ $2, DI
	DECQ CX
	JNZ  ai_bf16_sse2_bu_scale_scalar

ai_bf16_sse2_bu_done:
	RET
