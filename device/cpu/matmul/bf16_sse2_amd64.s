#include "textflag.h"

// func MatmulRowBF16SSE2Asm(cRow, aRow, b *uint16, inner, colsBlock, bCols int)
//
// Four output columns per block. Widen bf16→f32 via PUNPCKLWD+PSLLD, accumulate
// with MULPS/ADDPS, narrow with PSRLD+PEXTRW stores.
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
	VPXOR X3, X3, X3
	VPUNPCKLWD X3, X2, X2
	VPSLLD $16, X2, X2

	MULPS X2, X1
	ADDPS X1, X0

	ADDQ $2, R13
	ADDQ R8, R14
	DECQ CX
	JMP  mm_k_loop

mm_k_done:
	VPSRLD $16, X0, X0
	MOVL  X0, AX
	MOVW  AX, (DI)
	PSRLQ $32, X0
	MOVL  X0, AX
	MOVW  AX, 2(DI)
	PEXTRD $2, X0, AX
	MOVW  AX, 4(DI)
	PEXTRD $3, X0, AX
	MOVW  AX, 6(DI)

	ADDQ $8, DI
	ADDQ $8, BX
	SUBQ $4, DX
	JMP  mm_col_loop

mm_done:
	RET
