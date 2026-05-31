#include "textflag.h"
#include "../avx512_bf16_macros.inc"

// func MatmulRowBF16AVX512Asm(cRow, aRow, b *uint16, inner, colsBlock, bCols int)
TEXT ·MatmulRowBF16AVX512Asm(SB), NOSPLIT, $0-48
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
	JL   mm_col_w4

	VXORPS Y0, Y0, Y0

	MOVQ R12, CX
	MOVQ SI, R13
	MOVQ BX, R14

mm_k8:
	TESTQ CX, CX
	JZ    mm_k8_done

	MOVWLZX (R13), AX
	SHLQ  $16, AX
	VMOVD X1, AX
	VBROADCASTSS X1, Y1

	VMOVDQU Y2, (R14)
	VPMOVZXWD X2, Y3
	VPSLLD    $16, Y3, Y3

	VFMADD231PS Y1, Y3, Y0

	ADDQ $2, R13
	ADDQ R8, R14
	DECQ CX
	JMP  mm_k8

mm_k8_done:
	PACK_BF16_8H(DI)

	ADDQ $16, DI
	ADDQ $16, BX
	SUBQ $8, DX
	JMP  mm_col_loop

mm_col_w4:
	CMPQ DX, $4
	JL   mm_done

	XORPS X0, X0

	MOVQ R12, CX
	MOVQ SI, R13
	MOVQ BX, R14

mm_k4:
	TESTQ CX, CX
	JZ    mm_k4_done

	MOVWLZX (R13), AX
	SHLQ  $16, AX
	VMOVD X1, AX
	VBROADCASTSS X1, X1

	VMOVDQU X2, (R14)
	VPMOVZXWD X2, X3
	VPSLLD    $16, X3, X3

	VFMADD231PS X1, X3, X0

	ADDQ $2, R13
	ADDQ R8, R14
	DECQ CX
	JMP  mm_k4

mm_k4_done:
	VMOVAPS X0, X1
	PACK_BF16_X1_4H(DI)

	ADDQ $8, DI
	ADDQ $8, BX
	SUBQ $4, DX
	JMP  mm_col_loop

mm_done:
	RET
