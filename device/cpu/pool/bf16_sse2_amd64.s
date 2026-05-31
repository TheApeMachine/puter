#include "textflag.h"
#include "../sse2_bf16_macros.inc"

DATA poolNegInfBF16SSE2<>+0(SB)/4, $0xFF800000
DATA poolOneBF16SSE2<>+0(SB)/4, $0x3F800000
GLOBL poolNegInfBF16SSE2<>(SB), RODATA|NOPTR, $4
GLOBL poolOneBF16SSE2<>(SB), RODATA|NOPTR, $4

// func MaxPool2DStride1RowBF16SSE2Asm(
//     outRow, input *uint16,
//     outCols, kH, kW, inHStride, ihStart int,
// )
TEXT ·MaxPool2DStride1RowBF16SSE2Asm(SB), NOSPLIT, $0-56
	MOVQ outRow+0(FP), AX
	MOVQ input+8(FP), BX
	MOVQ outCols+16(FP), CX
	MOVQ kH+24(FP), DX
	MOVQ kW+32(FP), R8
	MOVQ inHStride+40(FP), R9
	MOVQ ihStart+48(FP), R10

	SHLQ $1, R9
	IMULQ R10, R9
	ADDQ R9, BX

	MOVSS poolNegInfBF16SSE2<>(SB), X0
	SHUFPS $0, X0, X0

mp_sse2_bf16_col_loop:
	CMPQ CX, $4
	JL   mp_sse2_bf16_done

	MOVAPS X0, X1

	MOVQ DX, R11
	MOVQ BX, R12

mp_sse2_bf16_kh_loop:
	TESTQ R11, R11
	JZ   mp_sse2_bf16_kh_done

	MOVQ R8, R13
	MOVQ R12, R14

mp_sse2_bf16_kw_loop:
	TESTQ R13, R13
	JZ   mp_sse2_bf16_kw_done

	BF16_LOAD_4H_TO_X4(R14, X2)
	VMAXPS  X2, X1, X1

	ADDQ $8, R14
	DECQ  R13
	JMP   mp_sse2_bf16_kw_loop

mp_sse2_bf16_kw_done:
	ADDQ R9, R12
	DECQ  R11
	JMP   mp_sse2_bf16_kh_loop

mp_sse2_bf16_kh_done:
	PACK_BF16_X1_4H(AX)

	ADDQ $8, AX
	ADDQ $8, BX
	SUBQ $4, CX
	JMP  mp_sse2_bf16_col_loop

mp_sse2_bf16_done:
	RET

// func AvgPool2DStride1RowBF16SSE2Asm(
//     outRow, input *uint16,
//     outCols, kH, kW, inHStride, ihStart int,
// )
TEXT ·AvgPool2DStride1RowBF16SSE2Asm(SB), NOSPLIT, $0-56
	MOVQ outRow+0(FP), AX
	MOVQ input+8(FP), BX
	MOVQ outCols+16(FP), CX
	MOVQ kH+24(FP), DX
	MOVQ kW+32(FP), R8
	MOVQ inHStride+40(FP), R9
	MOVQ ihStart+48(FP), R10

	SHLQ $1, R9
	IMULQ R10, R9
	ADDQ R9, BX

	IMULQ R8, DX
	MOVQ  DX, R11
	CVTSQ2SS R11, X14
	MOVSS poolOneBF16SSE2<>(SB), X13
	DIVSS X14, X13
	SHUFPS $0, X13, X0

	MOVQ kH+24(FP), DX
	MOVQ kW+32(FP), R8

ap_sse2_bf16_col_loop:
	CMPQ CX, $4
	JL   ap_sse2_bf16_done

	XORPS X1, X1

	MOVQ DX, R11
	MOVQ BX, R12

ap_sse2_bf16_kh_loop:
	TESTQ R11, R11
	JZ   ap_sse2_bf16_kh_done

	MOVQ R8, R13
	MOVQ R12, R14

ap_sse2_bf16_kw_loop:
	TESTQ R13, R13
	JZ   ap_sse2_bf16_kw_done

	BF16_LOAD_4H_TO_X4(R14, X2)
	ADDPS X2, X1

	ADDQ $8, R14
	DECQ  R13
	JMP  ap_sse2_bf16_kw_loop

ap_sse2_bf16_kw_done:
	ADDQ R9, R12
	DECQ  R11
	JMP  ap_sse2_bf16_kh_loop

ap_sse2_bf16_kh_done:
	MULPS X0, X1
	PACK_BF16_X1_4H(AX)

	ADDQ $8, AX
	ADDQ $8, BX
	SUBQ $4, CX
	JMP  ap_sse2_bf16_col_loop

ap_sse2_bf16_done:
	RET

DATA poolQuarterBF16SSE2<>+0(SB)/4, $0x3E800000
GLOBL poolQuarterBF16SSE2<>(SB), RODATA|NOPTR, $4

#define POOL22_SSE2_BF16_PAIR_MAX() \
	VSHUFPS $0xB1, X0, X0, X2; \
	VMAXPS  X0, X2, X3; \
	VSHUFPS $0xB1, X1, X1, X2; \
	VMAXPS  X1, X2, X4; \
	VMAXPS  X3, X4, X5; \
	VSHUFPS $0x88, X5, X5, X1

#define POOL22_SSE2_BF16_PAIR_SUM() \
	VSHUFPS $0xB1, X0, X0, X2; \
	VADDPS  X0, X2, X3; \
	VSHUFPS $0xB1, X1, X1, X2; \
	VADDPS  X1, X2, X4; \
	VADDPS  X3, X4, X5; \
	VSHUFPS $0x88, X5, X5, X1

// func MaxPool2x2Stride2RowBF16SSE2Asm(
//     outRow, input *uint16,
//     outCols, inWidth, ihStart int,
// )
TEXT ·MaxPool2x2Stride2RowBF16SSE2Asm(SB), NOSPLIT, $0-40
	MOVQ outRow+0(FP), AX
	MOVQ input+8(FP), BX
	MOVQ outCols+16(FP), CX
	MOVQ inWidth+24(FP), DX
	MOVQ ihStart+32(FP), R8

	SHLQ $1, DX, R9
	IMULQ R8, R9
	ADDQ R9, BX
	MOVQ BX, R10
	ADDQ R9, R10

mp22_sse2_bf16_col_loop:
	CMPQ CX, $2
	JL   mp22_sse2_bf16_done

	BF16_LOAD_4H_TO_X4(BX, X0)
	BF16_LOAD_4H_TO_X4(R10, X1)
	POOL22_SSE2_BF16_PAIR_MAX()
	PACK_BF16_X1_2H(AX)

	ADDQ $8, BX
	ADDQ $8, R10
	ADDQ $4, AX
	SUBQ $2, CX
	JMP  mp22_sse2_bf16_col_loop

mp22_sse2_bf16_done:
	RET

// func AvgPool2x2Stride2RowBF16SSE2Asm(
//     outRow, input *uint16,
//     outCols, inWidth, ihStart int,
// )
TEXT ·AvgPool2x2Stride2RowBF16SSE2Asm(SB), NOSPLIT, $0-40
	MOVQ outRow+0(FP), AX
	MOVQ input+8(FP), BX
	MOVQ outCols+16(FP), CX
	MOVQ inWidth+24(FP), DX
	MOVQ ihStart+32(FP), R8

	SHLQ $1, DX, R9
	IMULQ R8, R9
	ADDQ R9, BX
	MOVQ BX, R10
	ADDQ R9, R10

	VBROADCASTSS poolQuarterBF16SSE2<>(SB), X15

ap22_sse2_bf16_col_loop:
	CMPQ CX, $2
	JL   ap22_sse2_bf16_done

	BF16_LOAD_4H_TO_X4(BX, X0)
	BF16_LOAD_4H_TO_X4(R10, X1)
	POOL22_SSE2_BF16_PAIR_SUM()
	VMULPS  X15, X1, X1
	PACK_BF16_X1_2H(AX)

	ADDQ $8, BX
	ADDQ $8, R10
	ADDQ $4, AX
	SUBQ $2, CX
	JMP  ap22_sse2_bf16_col_loop

ap22_sse2_bf16_done:
	RET
