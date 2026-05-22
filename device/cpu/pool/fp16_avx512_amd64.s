#include "textflag.h"

#define VCVTPS2PH_Y0_X2 WORD $0xC4E3; WORD $0x7D1D; BYTE $0xD8; BYTE $0x00

DATA poolNegInfFP16AVX512<>+0(SB)/4, $0xFF800000
DATA poolOneFP16AVX512<>+0(SB)/4, $0x3F800000
DATA poolQuarterFP16AVX512<>+0(SB)/4, $0x3E800000
GLOBL poolNegInfFP16AVX512<>(SB), RODATA|NOPTR, $4
GLOBL poolOneFP16AVX512<>(SB), RODATA|NOPTR, $4
GLOBL poolQuarterFP16AVX512<>(SB), RODATA|NOPTR, $4

#define WIDEN_FP16_8H_TO_Y8(srcPtr, dstY) \
	VMOVDQU X2, (srcPtr); \
	VCVTPH2PS X2, dstY; \
	VMOVDQU X2, 8(srcPtr); \
	VCVTPH2PS X2, Y3; \
	VINSERTF128 $1, X3, dstY, dstY

#define NARROW_FP16_Y1_TO_4H(dstPtr) \
	VMOVAPS Y1, Y0; \
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

	VBROADCASTSS poolNegInfFP16AVX512<>(SB), Y0

mp512_fp16_col_loop:
	CMPQ CX, $4
	JL   mp512_fp16_done

	VMOVAPS Y0, Y1

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
	VCVTPH2PS X2, Y2
	VMAXPS  Y2, Y1, Y1

	ADDQ $8, R14
	DECQ  R13
	JMP   mp512_fp16_kw_loop

mp512_fp16_kw_done:
	ADDQ R9, R12
	DECQ  R11
	JMP   mp512_fp16_kh_loop

mp512_fp16_kh_done:
	NARROW_FP16_Y1_TO_4H(AX)

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
	VBROADCASTSS poolOneFP16AVX512<>(SB), X13
	VDIVSS  X14, X13, X12
	VBROADCASTSS X12, Y0

	MOVQ kH+24(FP), DX
	MOVQ kW+32(FP), R8

ap512_fp16_col_loop:
	CMPQ CX, $4
	JL   ap512_fp16_done

	VXORPS Y1, Y1, Y1

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
	VCVTPH2PS X2, Y2
	VADDPS  Y2, Y1, Y1

	ADDQ $8, R14
	DECQ  R13
	JMP   ap512_fp16_kw_loop

ap512_fp16_kw_done:
	ADDQ R9, R12
	DECQ  R11
	JMP   ap512_fp16_kh_loop

ap512_fp16_kh_done:
	VMULPS Y0, Y1, Y1
	NARROW_FP16_Y1_TO_4H(AX)

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

	WIDEN_FP16_8H_TO_Y8(BX, Y0)
	WIDEN_FP16_8H_TO_Y8(R10, Y1)

	VPERMILPS $0xB1, Y0, Y2
	VMAXPS    Y2, Y0, Y3

	VPERMILPS $0xB1, Y1, Y4
	VMAXPS    Y4, Y1, Y5

	VMAXPS    Y5, Y3, Y6

	VPERMILPS $0x88, Y6, Y1
	NARROW_FP16_Y1_TO_4H(AX)

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

	VBROADCASTSS poolQuarterFP16AVX512<>(SB), Y15

ap22512_fp16_col_loop:
	CMPQ CX, $4
	JL   ap22512_fp16_done

	WIDEN_FP16_8H_TO_Y8(BX, Y0)
	WIDEN_FP16_8H_TO_Y8(R10, Y1)

	VPERMILPS $0xB1, Y0, Y2
	VADDPS    Y2, Y0, Y3

	VPERMILPS $0xB1, Y1, Y4
	VADDPS    Y4, Y1, Y5

	VADDPS    Y5, Y3, Y6

	VPERMILPS $0x88, Y6, Y1
	VMULPS    Y15, Y1, Y1
	NARROW_FP16_Y1_TO_4H(AX)

	ADDQ $16, BX
	ADDQ $16, R10
	ADDQ $8, AX
	SUBQ $4, CX
	JMP  ap22512_fp16_col_loop

ap22512_fp16_done:
	RET
