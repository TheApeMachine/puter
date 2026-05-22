#include "textflag.h"

// func Conv2dStride1RowF32AVX2Asm(
//     outRow, input, weight *float32,
//     biasValue float32,
//     outCols, inChannels, kH, kW int,
//     inHStride, inCStride, wHStride, wCStride int,
//     ihStart, iwStart int,
// )
TEXT ·Conv2dStride1RowF32AVX2Asm(SB), NOSPLIT, $0-112
	MOVQ outRow+0(FP), DI
	MOVQ input+8(FP), SI
	MOVQ weight+16(FP), BX
	MOVSS biasValue+24(FP), X0
	VBROADCASTSS X0, Y0
	MOVQ outCols+32(FP), CX
	MOVQ inHStride+64(FP), R15
	MOVQ ihStart+96(FP), R11

	SHLQ $2, R15
	IMULQ R15, R11
	ADDQ R11, SI

col_block_loop:
	CMPQ CX, $4
	JL   conv2d_f32_avx2_done

	VMOVAPS Y0, Y1

	MOVQ inChannels+40(FP), R12
	MOVQ SI, R13
	MOVQ BX, R14

c_loop:
	TESTQ R12, R12
	JZ    c_done

	MOVQ kH+48(FP), R10
	MOVQ R13, R8
	MOVQ R14, R9

kh_loop:
	TESTQ R10, R10
	JZ    kh_done

	MOVQ kW+56(FP), AX
	MOVQ R8, R11
	MOVQ R9, R15

kw_loop:
	TESTQ AX, AX
	JZ    kw_done

	MOVSS (R15), X2
	VBROADCASTSS X2, Y2

	VMOVUPS (R11), Y3
	VFMADD231PS Y3, Y2, Y1

	ADDQ $4, R11
	ADDQ $4, R15
	DECQ AX
	JMP  kw_loop

kw_done:
	MOVQ inHStride+64(FP), R15
	SHLQ $2, R15
	ADDQ R15, R8
	MOVQ wHStride+80(FP), R15
	SHLQ $2, R15
	ADDQ R15, R9
	DECQ R10
	JMP  kh_loop

kh_done:
	MOVQ inCStride+72(FP), R15
	SHLQ $2, R15
	ADDQ R15, R13
	MOVQ wCStride+88(FP), R15
	SHLQ $2, R15
	ADDQ R15, R14
	DECQ R12
	JMP  c_loop

c_done:
	VMOVUPS Y1, (DI)

	ADDQ $16, DI
	ADDQ $16, SI
	SUBQ $4, CX
	JMP  col_block_loop

conv2d_f32_avx2_done:
	RET
