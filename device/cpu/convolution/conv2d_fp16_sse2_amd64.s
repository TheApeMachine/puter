#include "textflag.h"
#include "../f16c_fp16_macros.inc"

// func Conv2dStride1RowFP16SSE2Asm(
//     outRow, input, weight *uint16,
//     biasValue float32,
//     outCols, inChannels, kH, kW int,
//     inHStride, inCStride, wHStride, wCStride int,
//     ihStart, iwStart int,
// )
TEXT ·Conv2dStride1RowFP16SSE2Asm(SB), NOSPLIT, $0-112
	MOVQ outRow+0(FP), DI
	MOVQ input+8(FP), SI
	MOVQ weight+16(FP), BX
	MOVSS biasValue+24(FP), X0
	SHUFPS $0, X0, X0
	MOVQ outCols+32(FP), CX
	MOVQ inHStride+64(FP), R15
	MOVQ ihStart+96(FP), R11

	SHLQ $1, R15
	IMULQ R15, R11
	ADDQ R11, SI

col_block_loop:
	CMPQ CX, $4
	JL   conv2d_fp16_sse2_done

	MOVAPS X0, X1

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

	MOVWLZX (R15), DX
	VMOVD X2, DX
	VCVTPH2PS X2, X2
	SHUFPS $0, X2, X2

	VMOVDQU X3, (R11)
	VCVTPH2PS X3, X3
	MULPS X3, X2
	ADDPS X2, X1

	ADDQ $2, R11
	ADDQ $2, R15
	DECQ AX
	JMP  kw_loop

kw_done:
	MOVQ inHStride+64(FP), R15
	SHLQ $1, R15
	ADDQ R15, R8
	MOVQ wHStride+80(FP), R15
	SHLQ $1, R15
	ADDQ R15, R9
	DECQ R10
	JMP  kh_loop

kh_done:
	MOVQ inCStride+72(FP), R15
	SHLQ $1, R15
	ADDQ R15, R13
	MOVQ wCStride+88(FP), R15
	SHLQ $1, R15
	ADDQ R15, R14
	DECQ R12
	JMP  c_loop

c_done:
	FP16_NARROW_X1_TO_4H(DI)

	ADDQ $8, DI
	ADDQ $8, SI
	SUBQ $4, CX
	JMP  col_block_loop

conv2d_fp16_sse2_done:
	RET

// func Conv2dPatchDotFP16SSE2Asm(weight, patch *uint16, n int) float32
TEXT ·Conv2dPatchDotFP16SSE2Asm(SB), NOSPLIT, $0-28
	MOVQ weight+0(FP), SI
	MOVQ patch+8(FP), DI
	MOVQ n+16(FP), CX

	TESTQ CX, CX
	JZ    cpd_fp16_sse2_zero

	XORPS X0, X0

cpd_fp16_sse2_w4:
	CMPQ CX, $4
	JL   cpd_fp16_sse2_reduce

	VMOVDQU X1, (SI)
	VMOVDQU X2, (DI)
	VCVTPH2PS X1, X3
	VCVTPH2PS X2, X4
	MULPS     X4, X3
	ADDPS     X3, X0

	ADDQ $8, SI
	ADDQ $8, DI
	SUBQ $4, CX
	JMP  cpd_fp16_sse2_w4

cpd_fp16_sse2_reduce:
	MOVAPS X0, X1
	SHUFPS $2, X0, X1
	ADDPS  X1, X0
	MOVAPS X0, X1
	SHUFPS $1, X0, X1
	ADDPS  X1, X0

	TESTQ CX, CX
	JZ    cpd_fp16_sse2_store

cpd_fp16_sse2_tail:
	MOVWLZX (SI), AX
	VMOVD X1, AX
	VCVTPH2PS X1, X1
	MOVWLZX (DI), R8
	VMOVD X2, R8
	VCVTPH2PS X2, X2
	MULSS X2, X1
	ADDSS X1, X0

	ADDQ $2, SI
	ADDQ $2, DI
	DECQ CX
	JNZ  cpd_fp16_sse2_tail

cpd_fp16_sse2_store:
	MOVSS X0, ret+24(FP)
	RET

cpd_fp16_sse2_zero:
	XORPS X0, X0
	MOVSS X0, ret+24(FP)
	RET
