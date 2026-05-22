#include "textflag.h"

#define VCVTPS2PH_Y0_X2 WORD $0xC4E3; WORD $0x7D1D; BYTE $0xD8; BYTE $0x00
#define VCVTPS2PH_X0_X2 WORD $0xC4E3; WORD $0x7D1D; BYTE $0xD0; BYTE $0x00

// func MatmulRowFP16SSE2Asm(cRow, aRow, b *uint16, inner, colsBlock, bCols int)
TEXT ·MatmulRowFP16SSE2Asm(SB), NOSPLIT, $0-48
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
	VMOVD X1, AX
	VCVTPH2PS X1, X1
	SHUFPS $0, X1, X1

	MOVQ (R14), X2
	VCVTPH2PS X2, X2

	MULPS X2, X1
	ADDPS X1, X0

	ADDQ $2, R13
	ADDQ R8, R14
	DECQ CX
	JMP  mm_k_loop

mm_k_done:
	VCVTPS2PH_X0_X2
	MOVQ X2, AX
	MOVQ AX, (DI)

	ADDQ $8, DI
	ADDQ $8, BX
	SUBQ $4, DX
	JMP  mm_col_loop

mm_done:
	RET
