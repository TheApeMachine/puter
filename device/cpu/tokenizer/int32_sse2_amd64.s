#include "textflag.h"

// func TokenizerPackInt32SSE2Asm(dst, src *int32, count int)
TEXT ·TokenizerPackInt32SSE2Asm(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ count+16(FP), CX

tok_pack_sse2_w4:
	CMPQ CX, $4
	JL   tok_pack_sse2_tail

	VMOVDQU (SI), X0
	VMOVDQU X0, (DI)

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  tok_pack_sse2_w4

tok_pack_sse2_tail:
	TESTQ CX, CX
	JZ   tok_pack_sse2_done

tok_pack_sse2_scalar:
	MOVL (SI), AX
	MOVL AX, (DI)
	ADDQ $4, SI
	ADDQ $4, DI
	DECQ CX
	JNZ  tok_pack_sse2_scalar

tok_pack_sse2_done:
	RET
