#include "textflag.h"

// func MatmulRowBF16AVX512Asm(cRow, aRow, b *uint16, inner, colsBlock, bCols int)
//
// Eight output columns per block on AVX-512 (256-bit FMA path).
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
	VPSRLD $16, Y0, Y0
	VEXTRACTI128 $0, Y0, X2
	MOVL  X2, AX
	MOVW  AX, (DI)
	PSRLQ $32, X2
	MOVL  X2, AX
	MOVW  AX, 2(DI)
	PEXTRD $2, X2, AX
	MOVW  AX, 4(DI)
	PEXTRD $3, X2, AX
	MOVW  AX, 6(DI)

	VEXTRACTI128 $1, Y0, X2
	MOVL  X2, AX
	MOVW  AX, 8(DI)
	PSRLQ $32, X2
	MOVL  X2, AX
	MOVW  AX, 10(DI)
	PEXTRD $2, X2, AX
	MOVW  AX, 12(DI)
	PEXTRD $3, X2, AX
	MOVW  AX, 14(DI)

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
