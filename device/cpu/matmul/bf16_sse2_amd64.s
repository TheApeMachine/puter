#include "textflag.h"
#include "../sse2_bf16_macros.inc"

// func MatmulRowBF16SSE2Asm(cRow, aRow, b *uint16, inner, colsBlock, bCols int)
TEXT ·MatmulRowBF16SSE2Asm(SB), NOSPLIT, $0-48
	MOVQ cRow+0(FP), DI
	MOVQ aRow+8(FP), SI
	MOVQ b+16(FP), BX
	MOVQ inner+24(FP), R12
	MOVQ colsBlock+32(FP), DX
	MOVQ bCols+40(FP), R15

	MOVQ R15, R8
	SHLQ $1, R8

mm_col_loop:
	CMPQ DX, $4
	JL   mm_done

	XORPS X0, X0

	MOVQ R12, CX
	MOVQ SI, R13
	MOVQ BX, R14

mm_k_loop:
	TESTQ CX, CX
	JZ    mm_k_done

	MOVWLZX (R13), AX
	SHLQ  $16, AX
	MOVL  AX, X1
	SHUFPS $0, X1, X1

	MOVQ (R14), X2
	BF16_WIDEN_X2_LOW4(X2)

	MULPS X2, X1
	ADDPS X1, X0

	ADDQ $2, R13
	ADDQ R8, R14
	DECQ CX
	JMP  mm_k_loop

mm_k_done:
	PACK_BF16_ACCUM_X0_4H(DI)

	ADDQ $8, DI
	ADDQ $8, BX
	SUBQ $4, DX
	JMP  mm_col_loop

mm_done:
	RET
