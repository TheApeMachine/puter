#include "textflag.h"

// func TokenizerPackInt32AVX2Asm(dst, src *int32, count int)
TEXT ·TokenizerPackInt32AVX2Asm(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ count+16(FP), CX

tok_pack_avx2_w8:
	CMPQ CX, $8
	JL   tok_pack_avx2_w4

	VMOVDQU (SI), Y0
	VMOVDQU Y0, (DI)

	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $8, CX
	JMP  tok_pack_avx2_w8

tok_pack_avx2_w4:
	CMPQ CX, $4
	JL   tok_pack_avx2_tail

	VMOVDQU (SI), X0
	VMOVDQU X0, (DI)

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  tok_pack_avx2_w4

tok_pack_avx2_tail:
	TESTQ CX, CX
	JZ   tok_pack_avx2_done

tok_pack_avx2_scalar:
	MOVL (SI), AX
	MOVL AX, (DI)
	ADDQ $4, SI
	ADDQ $4, DI
	DECQ CX
	JNZ  tok_pack_avx2_scalar

tok_pack_avx2_done:
	RET
