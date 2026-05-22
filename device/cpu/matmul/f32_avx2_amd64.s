#include "textflag.h"

// func MatmulRowFloat32AVX2Asm(cRow, aRow, b *float32, inner, cols int)
TEXT ·MatmulRowFloat32AVX2Asm(SB), NOSPLIT, $0-40
	MOVQ cRow+0(FP), DI
	MOVQ aRow+8(FP), SI
	MOVQ b+16(FP), BX
	MOVQ inner+24(FP), R12
	MOVQ cols+32(FP), DX

	MOVQ DX, R8
	SHLQ $2, R8
	MOVQ R8, R10

mm_avx2_col_loop:
	TESTQ DX, DX
	JZ    mm_avx2_done

	CMPQ DX, $8
	JL    mm_avx2_col_w4

	VMOVUPS Y0, (DI)
	MOVQ  R12, CX
	MOVQ  SI, R13
	MOVQ  BX, R14

mm_avx2_k8:
	TESTQ CX, CX
	JZ    mm_avx2_k8_done

	MOVSS (R13), X1
	VBROADCASTSS X1, Y1
	VMOVUPS (R14), Y2
	VFMADD231PS Y0, Y2, Y1
	ADDQ  $4, R13
	ADDQ  R10, R14
	DECQ  CX
	JMP   mm_avx2_k8

mm_avx2_k8_done:
	VMOVUPS Y0, (DI)
	ADDQ  $32, DI
	ADDQ  $32, BX
	SUBQ  $8, DX
	JMP   mm_avx2_col_loop

mm_avx2_col_w4:
	TESTQ DX, DX
	JZ    mm_avx2_done

	CMPQ DX, $4
	JL    mm_avx2_col_tail

	VMOVUPS X0, (DI)
	MOVQ  R12, CX
	MOVQ  SI, R13
	MOVQ  BX, R14

mm_avx2_k4:
	TESTQ CX, CX
	JZ    mm_avx2_k4_done

	MOVSS (R13), X1
	VBROADCASTSS X1, X2
	VMOVUPS (R14), X3
	VFMADD231PS X0, X3, X2
	ADDQ  $4, R13
	ADDQ  R10, R14
	DECQ  CX
	JMP   mm_avx2_k4

mm_avx2_k4_done:
	VMOVUPS X0, (DI)
	ADDQ  $16, DI
	ADDQ  $16, BX
	SUBQ  $4, DX
	JMP   mm_avx2_col_loop

mm_avx2_col_tail:
	TESTQ DX, DX
	JZ    mm_avx2_done

	MOVQ DX, R11
	MOVQ DI, R14
	MOVQ BX, R15
	MOVQ SI, R13
	MOVQ R12, CX

mm_avx2_tail_col:
	MOVSS (R14), X0
	MOVQ  CX, R8
	MOVQ  R13, R9
	MOVQ  R15, R10

mm_avx2_tail_k:
	TESTQ R8, R8
	JZ    mm_avx2_tail_k_done

	MOVSS (R9), X1
	MOVSS (R10), X2
	MULSS X2, X1
	ADDSS X1, X0

	ADDQ $4, R9
	ADDQ R11, R10
	DECQ R8
	JMP  mm_avx2_tail_k

mm_avx2_tail_k_done:
	MOVSS X0, (R14)

	ADDQ $4, R14
	ADDQ $4, R15
	DECQ DX
	JMP  mm_avx2_col_tail

mm_avx2_done:
	RET
