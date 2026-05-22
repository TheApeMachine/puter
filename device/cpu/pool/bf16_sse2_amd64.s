#include "textflag.h"

DATA poolNegInfBF16SSE2<>+0(SB)/4, $0xFF800000
DATA poolOneBF16SSE2<>+0(SB)/4, $0x3F800000
GLOBL poolNegInfBF16SSE2<>(SB), RODATA|NOPTR, $4
GLOBL poolOneBF16SSE2<>(SB), RODATA|NOPTR, $4

#define WIDEN_BF16_4H_TO_X4(srcPtr, dstX) \
	VMOVDQU X2, (srcPtr); \
	VPMOVZXWD X2, dstX; \
	VPSLLD $16, dstX, dstX

#define NARROW_BF16_X1_TO_4H(dstPtr) \
	VPSRLD $16, X1, X1; \
	MOVL  X1, AX; \
	MOVW  AX, (dstPtr); \
	PEXTRD $1, X1, AX; \
	MOVW  AX, 2(dstPtr); \
	PEXTRD $2, X1, AX; \
	MOVW  AX, 4(dstPtr); \
	PEXTRD $3, X1, AX; \
	MOVW  AX, 6(dstPtr)

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

	WIDEN_BF16_4H_TO_X4(R14, X2)
	VMAXPS  X2, X1, X1

	ADDQ $8, R14
	DECQ  R13
	JMP   mp_sse2_bf16_kw_loop

mp_sse2_bf16_kw_done:
	ADDQ R9, R12
	DECQ  R11
	JMP   mp_sse2_bf16_kh_loop

mp_sse2_bf16_kh_done:
	NARROW_BF16_X1_TO_4H(AX)

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

	WIDEN_BF16_4H_TO_X4(R14, X2)
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
	NARROW_BF16_X1_TO_4H(AX)

	ADDQ $8, AX
	ADDQ $8, BX
	SUBQ $4, CX
	JMP  ap_sse2_bf16_col_loop

ap_sse2_bf16_done:
	RET
