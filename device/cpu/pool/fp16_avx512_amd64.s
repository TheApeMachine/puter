#include "textflag.h"
#include "../avx512_fp16_macros.inc"
#include "../f16c_fp16_macros.inc"

DATA poolNegInfHalfAVX512<>+0(SB)/2, $0xfc00
DATA poolOneF32AVX512<>+0(SB)/4, $0x3F800000
DATA poolQuarterHalfAVX512<>+0(SB)/2, $0x3400
GLOBL poolNegInfHalfAVX512<>(SB), RODATA|NOPTR, $2
GLOBL poolOneF32AVX512<>(SB), RODATA|NOPTR, $4
GLOBL poolQuarterHalfAVX512<>(SB), RODATA|NOPTR, $2

DATA poolFp16EvenShuf<>+0(SB)/1, $0x00
DATA poolFp16EvenShuf<>+1(SB)/1, $0x80
DATA poolFp16EvenShuf<>+2(SB)/1, $0x04
DATA poolFp16EvenShuf<>+3(SB)/1, $0x80
DATA poolFp16EvenShuf<>+4(SB)/1, $0x08
DATA poolFp16EvenShuf<>+5(SB)/1, $0x80
DATA poolFp16EvenShuf<>+6(SB)/1, $0x0c
DATA poolFp16EvenShuf<>+7(SB)/1, $0x80
DATA poolFp16EvenShuf<>+8(SB)/1, $0x80
DATA poolFp16EvenShuf<>+9(SB)/1, $0x80
DATA poolFp16EvenShuf<>+10(SB)/1, $0x80
DATA poolFp16EvenShuf<>+11(SB)/1, $0x80
DATA poolFp16EvenShuf<>+12(SB)/1, $0x80
DATA poolFp16EvenShuf<>+13(SB)/1, $0x80
DATA poolFp16EvenShuf<>+14(SB)/1, $0x80
DATA poolFp16EvenShuf<>+15(SB)/1, $0x80
DATA poolFp16OddShuf<>+0(SB)/1, $0x02
DATA poolFp16OddShuf<>+1(SB)/1, $0x80
DATA poolFp16OddShuf<>+2(SB)/1, $0x06
DATA poolFp16OddShuf<>+3(SB)/1, $0x80
DATA poolFp16OddShuf<>+4(SB)/1, $0x0a
DATA poolFp16OddShuf<>+5(SB)/1, $0x80
DATA poolFp16OddShuf<>+6(SB)/1, $0x0e
DATA poolFp16OddShuf<>+7(SB)/1, $0x80
DATA poolFp16OddShuf<>+8(SB)/1, $0x80
DATA poolFp16OddShuf<>+9(SB)/1, $0x80
DATA poolFp16OddShuf<>+10(SB)/1, $0x80
DATA poolFp16OddShuf<>+11(SB)/1, $0x80
DATA poolFp16OddShuf<>+12(SB)/1, $0x80
DATA poolFp16OddShuf<>+13(SB)/1, $0x80
DATA poolFp16OddShuf<>+14(SB)/1, $0x80
DATA poolFp16OddShuf<>+15(SB)/1, $0x80
GLOBL poolFp16EvenShuf<>(SB), RODATA|NOPTR, $16
GLOBL poolFp16OddShuf<>(SB), RODATA|NOPTR, $16

// func MaxPool2DStride1RowFP16AVX512Asm(
//     outRow, input *uint16,
//     outCols, kH, kW, inHStride, ihStart int,
// )
TEXT ·MaxPool2DStride1RowFP16AVX512Asm(SB), NOSPLIT, $0-56
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

	VPBROADCASTW poolNegInfHalfAVX512<>(SB), X1

mp512_fp16_col_loop:
	CMPQ CX, $4
	JL   mp512_fp16_done

	VMOVDQU X1, X0

	MOVQ DX, R11
	MOVQ BX, R12

mp512_fp16_kh_loop:
	TESTQ R11, R11
	JZ   mp512_fp16_kh_done

	MOVQ R8, R13
	MOVQ R12, R14

mp512_fp16_kw_loop:
	TESTQ R13, R13
	JZ   mp512_fp16_kw_done

	VMOVDQU X2, (R14)
	VMAXPH_X0_X2_X0

	ADDQ $8, R14
	DECQ  R13
	JMP   mp512_fp16_kw_loop

mp512_fp16_kw_done:
	ADDQ R9, R12
	DECQ  R11
	JMP   mp512_fp16_kh_loop

mp512_fp16_kh_done:
	MOVQ AX, DI
	STORE_X0_8H_DI

	ADDQ $8, AX
	ADDQ $8, BX
	SUBQ $4, CX
	JMP  mp512_fp16_col_loop

mp512_fp16_done:
	RET

// func AvgPool2DStride1RowFP16AVX512Asm(
//     outRow, input *uint16,
//     outCols, kH, kW, inHStride, ihStart int,
// )
TEXT ·AvgPool2DStride1RowFP16AVX512Asm(SB), NOSPLIT, $0-56
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
	IMULQ kH+24(FP), DX
	MOVQ  DX, R11
	CVTSQ2SS R11, X14
	VBROADCASTSS poolOneF32AVX512<>(SB), X13
	VDIVSS  X14, X13, X12
	VMOVAPS X12, X0
	VCVTPS2PH_X0_X2
	VPBROADCASTW X2, X3

	MOVQ kH+24(FP), DX
	MOVQ kW+32(FP), R8

ap512_fp16_col_loop:
	CMPQ CX, $4
	JL   ap512_fp16_done

	VPXORD X0, X0, X0

	MOVQ DX, R11
	MOVQ BX, R12

ap512_fp16_kh_loop:
	TESTQ R11, R11
	JZ   ap512_fp16_kh_done

	MOVQ R8, R13
	MOVQ R12, R14

ap512_fp16_kw_loop:
	TESTQ R13, R13
	JZ   ap512_fp16_kw_done

	VMOVDQU X2, (R14)
	VADDPH_X0_X2_X0

	ADDQ $8, R14
	DECQ  R13
	JMP   ap512_fp16_kw_loop

ap512_fp16_kw_done:
	ADDQ R9, R12
	DECQ  R11
	JMP   ap512_fp16_kh_loop

ap512_fp16_kh_done:
	VDIVPH_X0_X0_X3
	MOVQ AX, DI
	STORE_X0_8H_DI

	ADDQ $8, AX
	ADDQ $8, BX
	SUBQ $4, CX
	JMP  ap512_fp16_col_loop

ap512_fp16_done:
	RET

// func MaxPool2x2Stride2RowFP16AVX512Asm(
//     outRow, input *uint16,
//     outCols, inWidth, ihStart int,
// )
TEXT ·MaxPool2x2Stride2RowFP16AVX512Asm(SB), NOSPLIT, $0-40
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

mp22512_fp16_col_loop:
	CMPQ CX, $4
	JL   mp22512_fp16_done

	VMOVDQU X0, (BX)
	VMOVDQU X1, (R10)
	VPSHUFB poolFp16EvenShuf<>(SB), X0, X3
	VPSHUFB poolFp16OddShuf<>(SB), X0, X4
	VMAXPH_X5_X3_X4
	VPSHUFB poolFp16EvenShuf<>(SB), X1, X6
	VPSHUFB poolFp16OddShuf<>(SB), X1, X7
	VMAXPH_X8_X6_X7
	VMAXPH_X0_X5_X8
	MOVQ AX, DI
	STORE_X0_8H_DI

	ADDQ $16, BX
	ADDQ $16, R10
	ADDQ $8, AX
	SUBQ $4, CX
	JMP  mp22512_fp16_col_loop

mp22512_fp16_done:
	RET

// func AvgPool2x2Stride2RowFP16AVX512Asm(
//     outRow, input *uint16,
//     outCols, inWidth, ihStart int,
// )
TEXT ·AvgPool2x2Stride2RowFP16AVX512Asm(SB), NOSPLIT, $0-40
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

	VPBROADCASTW poolQuarterHalfAVX512<>(SB), X2

ap22512_fp16_col_loop:
	CMPQ CX, $4
	JL   ap22512_fp16_done

	VMOVDQU X0, (BX)
	VMOVDQU X1, (R10)
	VPSHUFB poolFp16EvenShuf<>(SB), X0, X3
	VPSHUFB poolFp16OddShuf<>(SB), X0, X4
	VADDPH_X5_X3_X4
	VPSHUFB poolFp16EvenShuf<>(SB), X1, X6
	VPSHUFB poolFp16OddShuf<>(SB), X1, X7
	VADDPH_X8_X6_X7
	VADDPH_X0_X5_X8
	VMULPH_X0_X2_X0
	MOVQ AX, DI
	STORE_X0_8H_DI

	ADDQ $16, BX
	ADDQ $16, R10
	ADDQ $8, AX
	SUBQ $4, CX
	JMP  ap22512_fp16_col_loop

ap22512_fp16_done:
	RET
