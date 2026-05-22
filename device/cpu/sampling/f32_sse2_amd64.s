#include "textflag.h"

// func GreedySampleFloat32SSE2Asm(logits *float32, count int) int32
TEXT ·GreedySampleFloat32SSE2Asm(SB), NOSPLIT, $0-20
	MOVQ logits+0(FP), SI
	MOVQ SI, BX
	MOVQ count+8(FP), CX

	TESTQ CX, CX
	JZ   greedy_sse2_zero

	CMPQ CX, $1
	JE   greedy_sse2_one

	MOVSS (SI), X0
	SHUFPS $0x00, X0, X0
	ADDQ $4, SI
	DECQ CX

greedy_sse2_max_w4:
	CMPQ CX, $4
	JL   greedy_sse2_max_tail

	MOVUPS (SI), X4
	MAXPS X4, X0

	ADDQ $16, SI
	SUBQ $4, CX
	JMP  greedy_sse2_max_w4

greedy_sse2_max_tail:
	TESTQ CX, CX
	JZ   greedy_sse2_max_done

greedy_sse2_max_scalar:
	MOVSS (SI), X4
	MAXSS X4, X0
	ADDQ  $4, SI
	DECQ  CX
	JNZ  greedy_sse2_max_scalar

greedy_sse2_max_done:
	MOVAPS X0, X4
	SHUFPS $0x4E, X0, X4
	MAXPS  X4, X0
	MOVAPS X0, X4
	SHUFPS $0xB1, X0, X4
	MAXPS  X4, X0
	MOVSS  X0, X0

	MOVQ BX, SI
	MOVQ count+8(FP), CX
	XORQ R8, R8

greedy_sse2_find_scalar:
	CMPQ R8, CX
	JGE  greedy_sse2_fail

	MOVSS (SI), X4
	UCOMISS X0, X4
	JNE  greedy_sse2_find_next
	MOVL R8, ret+16(FP)
	RET

greedy_sse2_find_next:
	ADDQ $4, SI
	INCQ R8
	JMP  greedy_sse2_find_scalar

greedy_sse2_fail:
	MOVQ count+8(FP), AX
	DECQ AX
	MOVL AX, ret+16(FP)
	RET

greedy_sse2_one:
	XORL AX, AX
	MOVL AX, ret+16(FP)
	RET

greedy_sse2_zero:
	XORL AX, AX
	MOVL AX, ret+16(FP)
	RET
