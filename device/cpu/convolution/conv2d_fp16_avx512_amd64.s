#include "textflag.h"

#define VCVTPS2PH_Y0_X2 WORD $0xC4E3; WORD $0x7D1D; BYTE $0xD8; BYTE $0x00

#define WIDEN_FP16_4H(srcReg, dstY) \
	VMOVDQU X2, (srcReg); \
	VCVTPH2PS X2, dstY

#define NARROW_FP16_Y0_TO_4H(dstReg) \
	VCVTPS2PH_Y0_X2; \
	VMOVDQU X2, (dstReg)

// func Conv2dStride1RowFP16AVX512Asm(
//     outRow, input, weight *uint16,
//     biasValue float32,
//     outCols, inChannels, kH, kW int,
//     inHStride, inCStride, wHStride, wCStride int,
//     ihStart, iwStart int,
// )
TEXT ·Conv2dStride1RowFP16AVX512Asm(SB), NOSPLIT, $0-112
	MOVQ outRow+0(FP), DI
	MOVQ input+8(FP), SI
	MOVQ weight+16(FP), BX
	MOVSS biasValue+24(FP), X0
	VBROADCASTSS X0, Y0
	MOVQ outCols+32(FP), CX
	MOVQ inHStride+64(FP), R15
	MOVQ ihStart+96(FP), R11

	SHLQ $1, R15
	IMULQ R15, R11
	ADDQ R11, SI

col_block_loop:
	CMPQ CX, $4
	JL   conv2d_fp16_done

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

	MOVWLZX (R15), DX
	VMOVD X2, DX
	VCVTPH2PS X2, X2
	VBROADCASTSS X2, Y2

	WIDEN_FP16_4H(R11, Y3)
	VFMADD231PS Y1, Y3, Y2

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
	VMOVAPS Y1, Y0
	NARROW_FP16_Y0_TO_4H(DI)

	ADDQ $8, DI
	ADDQ $8, SI
	SUBQ $4, CX
	JMP  col_block_loop

conv2d_fp16_done:
	RET

// func Conv2dPatchDotFP16AVX512Asm(weight, patch *uint16, n int) float32
TEXT ·Conv2dPatchDotFP16AVX512Asm(SB), NOSPLIT, $0-28
	MOVQ weight+0(FP), SI
	MOVQ patch+8(FP), DI
	MOVQ n+16(FP), CX

	TESTQ CX, CX
	JZ    cpd_fp16_zero

	VXORPS Y0, Y0, Y0

cpd_fp16_w8:
	CMPQ CX, $8
	JL   cpd_fp16_w4

	VMOVDQU X1, (SI)
	VMOVDQU X2, (DI)
	VCVTPH2PS X1, Y3
	VCVTPH2PS X2, Y4
	VMULPS    Y4, Y3, Y3
	VADDPS    Y3, Y0, Y0

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $8, CX
	JMP  cpd_fp16_w8

cpd_fp16_w4:
	CMPQ CX, $4
	JL   cpd_fp16_reduce

	VMOVDQU X1, (SI)
	VMOVDQU X2, (DI)
	VCVTPH2PS X1, Y3
	VCVTPH2PS X2, Y4
	VMULPS    Y4, Y3, Y3
	VADDPS    Y3, Y0, Y0

	ADDQ $8, SI
	ADDQ $8, DI
	SUBQ $4, CX
	JMP  cpd_fp16_w4

cpd_fp16_reduce:
	VEXTRACTF128 $1, Y0, X1
	VADDPS       X1, X0, X0
	VHADDPS      X0, X0, X0
	VHADDPS      X0, X0, X0

	TESTQ CX, CX
	JZ    cpd_fp16_store

cpd_fp16_tail:
	MOVWLZX (SI), AX
	VMOVD X1, AX
	VCVTPH2PS X1, X1
	MOVWLZX (DI), R8
	VMOVD X2, R8
	VCVTPH2PS X2, X2
	VMULSS X2, X1, X1
	VADDSS X1, X0, X0

	ADDQ $2, SI
	ADDQ $2, DI
	DECQ CX
	JNZ  cpd_fp16_tail

cpd_fp16_store:
	MOVSS X0, ret+24(FP)
	RET

cpd_fp16_zero:
	XORPS X0, X0
	MOVSS X0, ret+24(FP)
	RET
