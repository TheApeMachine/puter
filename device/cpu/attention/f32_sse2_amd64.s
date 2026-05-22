#include "textflag.h"

// func FlashAttentionOnlineUpdateSSE2Asm(
//     acc, valueRow *float32,
//     alpha, shifted float32,
//     n int,
// )
//
// acc[i] = acc[i]*alpha + valueRow[i]*shifted for i in [0,n).
TEXT ·FlashAttentionOnlineUpdateSSE2Asm(SB), NOSPLIT, $0-32
	MOVQ acc+0(FP), SI
	MOVQ valueRow+8(FP), DI
	MOVQ n+24(FP), CX

	MOVSS alpha+16(FP), X14
	SHUFPS $0, X14, X14
	MOVSS shifted+20(FP), X15
	SHUFPS $0, X15, X15

flash_upd_sse2_w4:
	CMPQ CX, $4
	JL   flash_upd_sse2_tail

	MOVUPS (SI), X0
	MOVUPS (DI), X2
	MULPS  X14, X0
	MULPS  X15, X2
	ADDPS  X2, X0
	MOVUPS X0, (SI)

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  flash_upd_sse2_w4

flash_upd_sse2_tail:
	TESTQ CX, CX
	JZ   flash_upd_sse2_done

flash_upd_sse2_scalar:
	MOVSS (SI), X0
	MOVSS (DI), X2
	MULSS X14, X0
	MULSS X15, X2
	ADDSS X2, X0
	MOVSS X0, (SI)
	ADDQ  $4, SI
	ADDQ  $4, DI
	DECQ  CX
	JNZ  flash_upd_sse2_scalar

flash_upd_sse2_done:
	RET

// func FlashAttentionScaleSSE2Asm(
//     out, acc *float32,
//     invNormalizer float32,
//     n int,
// )
//
// out[i] = acc[i] * invNormalizer for i in [0,n).
TEXT ·FlashAttentionScaleSSE2Asm(SB), NOSPLIT, $0-32
	MOVQ out+0(FP), DI
	MOVQ acc+8(FP), SI
	MOVQ n+24(FP), CX

	MOVSS invNormalizer+16(FP), X15
	SHUFPS $0, X15, X15

flash_scale_sse2_w4:
	CMPQ CX, $4
	JL   flash_scale_sse2_tail

	MOVUPS (SI), X0
	MULPS  X15, X0
	MOVUPS X0, (DI)

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  flash_scale_sse2_w4

flash_scale_sse2_tail:
	TESTQ CX, CX
	JZ   flash_scale_sse2_done

flash_scale_sse2_scalar:
	MOVSS (SI), X0
	MULSS X15, X0
	MOVSS X0, (DI)
	ADDQ  $4, SI
	ADDQ  $4, DI
	DECQ  CX
	JNZ  flash_scale_sse2_scalar

flash_scale_sse2_done:
	RET
