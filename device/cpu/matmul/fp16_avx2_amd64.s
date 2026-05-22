#include "textflag.h"

#define VCVTPS2PH_Y0_X2 WORD $0xC4E3; WORD $0x7D1D; BYTE $0xD8; BYTE $0x00

// func MatmulRowFP16AVX2Asm(cRow, aRow, b *uint16, inner, colsBlock, bCols int)
TEXT ·MatmulRowFP16AVX2Asm(SB), NOSPLIT, $0-48
	MOVQ cRow+0(FP), DI
	MOVQ aRow+8(FP), SI
	MOVQ b+16(FP), BX
	MOVQ inner+24(FP), R12
	MOVQ colsBlock+32(FP), DX
	MOVQ bCols+40(FP), R15

	MOVQ R15, R8
	SHLQ $1, R8

mm_col_loop:
	CMPQ DX, $8
	JL   mm_done

	VXORPS Y0, Y0, Y0

	MOVQ R12, CX
	MOVQ SI, R13
	MOVQ BX, R14

mm_k_loop:
	TESTQ CX, CX
	JZ    mm_k_done

	MOVWLZX (R13), AX
	VMOVD X1, AX
	VCVTPH2PS X1, X1
	VBROADCASTSS X1, Y1

	VMOVDQU X2, (R14)
	VCVTPH2PS X2, Y2

	VFMADD231PS Y0, Y2, Y1

	ADDQ $2, R13
	ADDQ R8, R14
	DECQ CX
	JMP  mm_k_loop

mm_k_done:
	VCVTPS2PH_Y0_X2
	VMOVDQU X2, (DI)

	ADDQ $16, DI
	ADDQ $16, BX
	SUBQ $8, DX
	JMP  mm_col_loop

mm_done:
	RET
