#include "textflag.h"

#define VCVTPS2PH_Y0_X2 WORD $0xC4E3; WORD $0x7D1D; BYTE $0xD8; BYTE $0x00

DATA poolNegInfHalfAVX512<>+0(SB)/2, $0xfc00
DATA poolOneHalfAVX512<>+0(SB)/2, $0x3c00
DATA poolQuarterHalfAVX512<>+0(SB)/2, $0x3400
GLOBL poolNegInfHalfAVX512<>(SB), RODATA|NOPTR, $2
GLOBL poolOneHalfAVX512<>(SB), RODATA|NOPTR, $2
GLOBL poolQuarterHalfAVX512<>(SB), RODATA|NOPTR, $2

#define NARROW_FP16_X1_TO_4H(dstPtr) \
	VCVTPS2PH_Y0_X2; \
	VMOVDQU X2, (dstPtr)

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

	VPBROADCASTW poolNegInfHalfAVX512<>(SB), X0

mp512_fp16_col_loop:
	CMPQ CX, $4
	JL   mp512_fp16_done

	VMOVDQU X0, X1

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
	VMAXPH X2, X1, X1

	ADDQ $8, R14
	DECQ  R13
	JMP   mp512_fp16_kw_loop

mp512_fp16_kw_done:
	ADDQ R9, R12
	DECQ  R11
	JMP   mp512_fp16_kh_loop

mp512_fp16_kh_done:
	VMOVDQU X1, (AX)

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
	MOVQ  DX, R11
	CVTSQ2SS R11, X14
	MOVSS X14, X0
	VCVTPS2PH_Y0_X2
	VPBROADCASTW X2, X15
	VPBROADCASTW poolOneHalfAVX512<>(SB), X14
	VDIVPH X15, X14, X0

	MOVQ kH+24(FP), DX
	MOVQ kW+32(FP), R8

ap512_fp16_col_loop:
	CMPQ CX, $4
	JL   ap512_fp16_done

	VPXORD X1, X1, X1

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
	VADDPH X2, X1, X1

	ADDQ $8, R14
	DECQ  R13
	JMP   ap512_fp16_kw_loop

ap512_fp16_kw_done:
	ADDQ R9, R12
	DECQ  R11
	JMP   ap512_fp16_kh_loop

ap512_fp16_kh_done:
	VMULPH X0, X1, X1
	VMOVDQU X1, (AX)

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

	VMOVDQU (BX), X0
	VMOVDQU 8(BX), X1
	VMAXPH X1, X0, X2
	VMOVDQU (R10), X3
	VMOVDQU 8(R10), X4
	VMAXPH X4, X3, X5
	VMAXPH X5, X2, X0
	VMOVDQU X0, (AX)

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

	VPBROADCASTW poolQuarterHalfAVX512<>(SB), X15

	SHLQ $1, DX, R9
	IMULQ R8, R9
	ADDQ R9, BX
	MOVQ BX, R10
	ADDQ R9, R10

ap22512_fp16_col_loop:
	CMPQ CX, $4
	JL   ap22512_fp16_done

	VMOVDQU (BX), X0
	VMOVDQU 8(BX), X1
	VADDPH X1, X0, X2
	VMOVDQU (R10), X3
	VMOVDQU 8(R10), X4
	VADDPH X4, X3, X5
	VADDPH X5, X2, X0
	VMULPH X15, X0, X0
	VMOVDQU X0, (AX)

	ADDQ $16, BX
	ADDQ $16, R10
	ADDQ $8, AX
	SUBQ $4, CX
	JMP  ap22512_fp16_col_loop

ap22512_fp16_done:
	RET
