#include "textflag.h"

#define VCVTPS2PH_X0_X2 WORD $0xC4E3; WORD $0x7D1D; BYTE $0xD0; BYTE $0x00

DATA poolNegInfFP16SSE2<>+0(SB)/4, $0xFF800000
DATA poolOneFP16SSE2<>+0(SB)/4, $0x3F800000
GLOBL poolNegInfFP16SSE2<>(SB), RODATA|NOPTR, $4
GLOBL poolOneFP16SSE2<>(SB), RODATA|NOPTR, $4

#define NARROW_FP16_X1_TO_4H(dstPtr) \
	MOVAPS X1, X0; \
	VCVTPS2PH_X0_X2; \
	VMOVDQU X2, (dstPtr)

// func MaxPool2DStride1RowFP16SSE2Asm(
//     outRow, input *uint16,
//     outCols, kH, kW, inHStride, ihStart int,
// )
TEXT ·MaxPool2DStride1RowFP16SSE2Asm(SB), NOSPLIT, $0-56
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

	MOVSS poolNegInfFP16SSE2<>(SB), X0
	SHUFPS $0, X0, X0

mp_sse2_fp16_col_loop:
	CMPQ CX, $4
	JL   mp_sse2_fp16_done

	MOVAPS X0, X1

	MOVQ DX, R11
	MOVQ BX, R12

mp_sse2_fp16_kh_loop:
	TESTQ R11, R11
	JZ   mp_sse2_fp16_kh_done

	MOVQ R8, R13
	MOVQ R12, R14

mp_sse2_fp16_kw_loop:
	TESTQ R13, R13
	JZ   mp_sse2_fp16_kw_done

	VMOVDQU X2, (R14)
	VCVTPH2PS X2, X2
	VMAXPS  X2, X1, X1

	ADDQ $8, R14
	DECQ  R13
	JMP  mp_sse2_fp16_kw_loop

mp_sse2_fp16_kw_done:
	ADDQ R9, R12
	DECQ  R11
	JMP  mp_sse2_fp16_kh_loop

mp_sse2_fp16_kh_done:
	NARROW_FP16_X1_TO_4H(AX)

	ADDQ $8, AX
	ADDQ $8, BX
	SUBQ $4, CX
	JMP  mp_sse2_fp16_col_loop

mp_sse2_fp16_done:
	RET

// func AvgPool2DStride1RowFP16SSE2Asm(
//     outRow, input *uint16,
//     outCols, kH, kW, inHStride, ihStart int,
// )
TEXT ·AvgPool2DStride1RowFP16SSE2Asm(SB), NOSPLIT, $0-56
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
	MOVSS poolOneFP16SSE2<>(SB), X13
	DIVSS X14, X13
	SHUFPS $0, X13, X0

	MOVQ kH+24(FP), DX
	MOVQ kW+32(FP), R8

ap_sse2_fp16_col_loop:
	CMPQ CX, $4
	JL   ap_sse2_fp16_done

	XORPS X1, X1

	MOVQ DX, R11
	MOVQ BX, R12

ap_sse2_fp16_kh_loop:
	TESTQ R11, R11
	JZ   ap_sse2_fp16_kh_done

	MOVQ R8, R13
	MOVQ R12, R14

ap_sse2_fp16_kw_loop:
	TESTQ R13, R13
	JZ   ap_sse2_fp16_kw_done

	VMOVDQU X2, (R14)
	VCVTPH2PS X2, X2
	ADDPS X2, X1

	ADDQ $8, R14
	DECQ  R13
	JMP  ap_sse2_fp16_kw_loop

ap_sse2_fp16_kw_done:
	ADDQ R9, R12
	DECQ  R11
	JMP  ap_sse2_fp16_kh_loop

ap_sse2_fp16_kh_done:
	MULPS X0, X1
	NARROW_FP16_X1_TO_4H(AX)

	ADDQ $8, AX
	ADDQ $8, BX
	SUBQ $4, CX
	JMP  ap_sse2_fp16_col_loop

ap_sse2_fp16_done:
	RET

DATA poolQuarterFP16SSE2<>+0(SB)/4, $0x3E800000
GLOBL poolQuarterFP16SSE2<>(SB), RODATA|NOPTR, $4

#define WIDEN_FP16_4H_TO_X4(srcPtr, dstX) \
	VMOVDQU X2, (srcPtr); \
	VCVTPH2PS X2, dstX

#define NARROW_FP16_X1_TO_2H(dstPtr) \
	MOVAPS X1, X0; \
	VCVTPS2PH_X0_X2; \
	MOVL  X2, AX; \
	MOVW  AX, (dstPtr); \
	PEXTRW $1, X2, AX; \
	MOVW  AX, 2(dstPtr)

#define POOL22_SSE2_FP16_PAIR_MAX() \
	VSHUFPS $0xB1, X0, X0, X2; \
	VMAXPS  X0, X2, X3; \
	VSHUFPS $0xB1, X1, X1, X2; \
	VMAXPS  X1, X2, X4; \
	VMAXPS  X3, X4, X5; \
	VSHUFPS $0x88, X5, X5, X1

#define POOL22_SSE2_FP16_PAIR_SUM() \
	VSHUFPS $0xB1, X0, X0, X2; \
	VADDPS  X0, X2, X3; \
	VSHUFPS $0xB1, X1, X1, X2; \
	VADDPS  X1, X2, X4; \
	VADDPS  X3, X4, X5; \
	VSHUFPS $0x88, X5, X5, X1

// func MaxPool2x2Stride2RowFP16SSE2Asm(
//     outRow, input *uint16,
//     outCols, inWidth, ihStart int,
// )
TEXT ·MaxPool2x2Stride2RowFP16SSE2Asm(SB), NOSPLIT, $0-40
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

mp22_sse2_fp16_col_loop:
	CMPQ CX, $2
	JL   mp22_sse2_fp16_done

	WIDEN_FP16_4H_TO_X4(BX, X0)
	WIDEN_FP16_4H_TO_X4(R10, X1)
	POOL22_SSE2_FP16_PAIR_MAX()
	NARROW_FP16_X1_TO_2H(AX)

	ADDQ $8, BX
	ADDQ $8, R10
	ADDQ $4, AX
	SUBQ $2, CX
	JMP  mp22_sse2_fp16_col_loop

mp22_sse2_fp16_done:
	RET

// func AvgPool2x2Stride2RowFP16SSE2Asm(
//     outRow, input *uint16,
//     outCols, inWidth, ihStart int,
// )
TEXT ·AvgPool2x2Stride2RowFP16SSE2Asm(SB), NOSPLIT, $0-40
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

	VBROADCASTSS poolQuarterFP16SSE2<>(SB), X15

ap22_sse2_fp16_col_loop:
	CMPQ CX, $2
	JL   ap22_sse2_fp16_done

	WIDEN_FP16_4H_TO_X4(BX, X0)
	WIDEN_FP16_4H_TO_X4(R10, X1)
	POOL22_SSE2_FP16_PAIR_SUM()
	VMULPS  X15, X1, X1
	NARROW_FP16_X1_TO_2H(AX)

	ADDQ $8, BX
	ADDQ $8, R10
	ADDQ $4, AX
	SUBQ $2, CX
	JMP  ap22_sse2_fp16_col_loop

ap22_sse2_fp16_done:
	RET
