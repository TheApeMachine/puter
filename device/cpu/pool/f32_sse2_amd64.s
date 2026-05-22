#include "textflag.h"

DATA poolNegInfSSE2<>+0(SB)/4, $0xFF800000
DATA poolOneSSE2<>+0(SB)/4, $0x3F800000
DATA poolQuarterSSE2<>+0(SB)/4, $0x3E800000
GLOBL poolNegInfSSE2<>(SB), RODATA|NOPTR, $4
GLOBL poolOneSSE2<>(SB), RODATA|NOPTR, $4
GLOBL poolQuarterSSE2<>(SB), RODATA|NOPTR, $4

// func MaxPool2DStride1RowSSE2Asm(
//     outRow *float32, input *float32,
//     outCols, kH, kW int,
//     inHStride int,
//     ihStart int,
// )
TEXT ·MaxPool2DStride1RowSSE2Asm(SB), NOSPLIT, $0-56
	MOVQ outRow+0(FP), AX
	MOVQ input+8(FP), BX
	MOVQ outCols+16(FP), CX
	MOVQ kH+24(FP), DX
	MOVQ kW+32(FP), R8
	MOVQ inHStride+40(FP), R9
	MOVQ ihStart+48(FP), R10

	SHLQ $2, R9
	IMULQ R10, R9
	ADDQ R9, BX

	MOVSS poolNegInfSSE2<>(SB), X0
	SHUFPS $0, X0, X0

mp_sse2_col_loop:
	CMPQ CX, $4
	JL   mp_sse2_done

	MOVAPS X0, X1

	MOVQ DX, R11
	MOVQ BX, R12

mp_sse2_kh_loop:
	TESTQ R11, R11
	JZ   mp_sse2_kh_done

	MOVQ R8, R13
	MOVQ R12, R14

mp_sse2_kw_loop:
	TESTQ R13, R13
	JZ   mp_sse2_kw_done

	MOVUPS (R14), X2
	MAXPS  X2, X1

	ADDQ $4, R14
	DECQ  R13
	JMP   mp_sse2_kw_loop

mp_sse2_kw_done:
	ADDQ R9, R12
	DECQ  R11
	JMP  mp_sse2_kh_loop

mp_sse2_kh_done:
	MOVUPS X1, (AX)

	ADDQ $16, AX
	ADDQ $16, BX
	SUBQ $4, CX
	JMP  mp_sse2_col_loop

mp_sse2_done:
	RET

// func AvgPool2DStride1RowSSE2Asm(
//     outRow *float32, input *float32,
//     outCols, kH, kW int,
//     inHStride int,
//     ihStart int,
// )
TEXT ·AvgPool2DStride1RowSSE2Asm(SB), NOSPLIT, $0-56
	MOVQ outRow+0(FP), AX
	MOVQ input+8(FP), BX
	MOVQ outCols+16(FP), CX
	MOVQ kH+24(FP), DX
	MOVQ kW+32(FP), R8
	MOVQ inHStride+40(FP), R9
	MOVQ ihStart+48(FP), R10

	SHLQ $2, R9
	IMULQ R10, R9
	ADDQ R9, BX

	IMULQ R8, DX
	MOVQ  DX, R15
	CVTSQ2SS R15, X14
	MOVSS poolOneSSE2<>(SB), X13
	DIVSS X14, X13
	SHUFPS $0, X13, X0

	MOVQ kH+24(FP), DX
	MOVQ kW+32(FP), R8

ap_sse2_col_loop:
	CMPQ CX, $4
	JL   ap_sse2_done

	XORPS X1, X1

	MOVQ DX, R11
	MOVQ BX, R12

ap_sse2_kh_loop:
	TESTQ R11, R11
	JZ   ap_sse2_kh_done

	MOVQ R8, R13
	MOVQ R12, R14

ap_sse2_kw_loop:
	TESTQ R13, R13
	JZ   ap_sse2_kw_done

	MOVUPS (R14), X2
	ADDPS  X2, X1

	ADDQ $4, R14
	DECQ  R13
	JMP   ap_sse2_kw_loop

ap_sse2_kw_done:
	ADDQ R9, R12
	DECQ  R11
	JMP  ap_sse2_kh_loop

ap_sse2_kh_done:
	MULPS X0, X1
	MOVUPS X1, (AX)

	ADDQ $16, AX
	ADDQ $16, BX
	SUBQ $4, CX
	JMP  ap_sse2_col_loop

ap_sse2_done:
	RET

// func MaxPool2x2Stride2RowSSE2Asm(
//     outRow *float32, input *float32,
//     outCols, inWidth, ihStart int,
// )
TEXT ·MaxPool2x2Stride2RowSSE2Asm(SB), NOSPLIT, $0-40
	MOVQ outRow+0(FP), AX
	MOVQ input+8(FP), BX
	MOVQ outCols+16(FP), CX
	MOVQ inWidth+24(FP), DX
	MOVQ ihStart+32(FP), R8

	SHLQ $2, DX, R9
	IMULQ R8, R9
	ADDQ R9, BX
	MOVQ BX, R10
	ADDQ R9, R10

mp22_sse2_col_loop:
	CMPQ CX, $4
	JL   mp22_sse2_done

	MOVUPS (BX), X0
	MOVUPS (R10), X1

	MOVAPS X0, X2
	SHUFPS $0xB1, X0, X2
	MAXPS  X2, X0

	MOVAPS X1, X2
	SHUFPS $0xB1, X1, X2
	MAXPS  X2, X1

	MAXPS  X1, X0

	MOVAPS X0, X2
	SHUFPS $0x88, X0, X2
	MOVUPS X2, (AX)

	ADDQ $32, BX
	ADDQ $32, R10
	ADDQ $16, AX
	SUBQ $4, CX
	JMP  mp22_sse2_col_loop

mp22_sse2_done:
	RET

// func AvgPool2x2Stride2RowSSE2Asm(
//     outRow *float32, input *float32,
//     outCols, inWidth, ihStart int,
// )
TEXT ·AvgPool2x2Stride2RowSSE2Asm(SB), NOSPLIT, $0-40
	MOVQ outRow+0(FP), AX
	MOVQ input+8(FP), BX
	MOVQ outCols+16(FP), CX
	MOVQ inWidth+24(FP), DX
	MOVQ ihStart+32(FP), R8

	SHLQ $2, DX, R9
	IMULQ R8, R9
	ADDQ R9, BX
	MOVQ BX, R10
	ADDQ R9, R10

	MOVSS poolQuarterSSE2<>(SB), X15
	SHUFPS $0, X15, X15

ap22_sse2_col_loop:
	CMPQ CX, $4
	JL   ap22_sse2_done

	MOVUPS (BX), X0
	MOVUPS (R10), X1

	MOVAPS X0, X2
	SHUFPS $0xB1, X0, X2
	ADDPS  X2, X0

	MOVAPS X1, X2
	SHUFPS $0xB1, X1, X2
	ADDPS  X2, X1

	ADDPS  X1, X0

	MOVAPS X0, X2
	SHUFPS $0x88, X0, X2
	MULPS  X15, X2
	MOVUPS X2, (AX)

	ADDQ $32, BX
	ADDQ $32, R10
	ADDQ $16, AX
	SUBQ $4, CX
	JMP  ap22_sse2_col_loop

ap22_sse2_done:
	RET
