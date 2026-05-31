#include "textflag.h"
#include "../avx512_fp16_macros.inc"
#include "../f16c_fp16_macros.inc"

#define ACCUM_FP16_SUM_Y10 \
	VEXTRACTI128 $0, Y10, X10; \
	VEXTRACTI128 $1, Y10, X11; \
	VCVTPH2PS X10, Y12; \
	VCVTPH2PS X11, Y13; \
	VADDPS Y12, Y0, Y0; \
	VADDPS Y13, Y0, Y0

// func SumFloat16AVX512Asm(src *uint16, count int) uint16
TEXT ·SumFloat16AVX512Asm(SB), NOSPLIT, $0-18
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX

	TESTQ CX, CX
	JZ    sum_fp16_zero

	VXORPS Y0, Y0, Y0

sum_fp16_w16:
	CMPQ CX, $16
	JL   sum_fp16_w8

	VMOVUPH_Y0_SI
	VPXORD Y10, Y10, Y10
	VADDPH_Y10_Y0_Y10
	ACCUM_FP16_SUM_Y10

	ADDQ $16, SI

	VMOVUPH_Y0_SI
	VPXORD Y10, Y10, Y10
	VADDPH_Y10_Y0_Y10
	ACCUM_FP16_SUM_Y10

	ADDQ $16, SI
	SUBQ $16, CX
	JMP  sum_fp16_w16

sum_fp16_w8:
	CMPQ CX, $8
	JL   sum_fp16_w4

	VMOVUPH_Y0_SI
	VPXORD Y10, Y10, Y10
	VADDPH_Y10_Y0_Y10
	ACCUM_FP16_SUM_Y10

	ADDQ $16, SI
	SUBQ $8, CX
	JMP  sum_fp16_w8

sum_fp16_w4:
	CMPQ CX, $4
	JL   sum_fp16_reduce

	VMOVDQU X1, (SI)
	VPXORD X10, X10, X10
	VADDPH_X10_X1_X10
	VCVTPH2PS X10, X12
	VADDPS X12, X0, X0

	ADDQ $8, SI
	SUBQ $4, CX
	JMP  sum_fp16_w4

sum_fp16_reduce:
	VEXTRACTF128 $1, Y0, X1
	VADDPS       X1, X0, X0
	VHADDPS      X0, X0, X0
	VHADDPS      X0, X0, X0

	TESTQ CX, CX
	JZ    sum_fp16_store

sum_fp16_scalar:
	MOVWLZX (SI), AX
	VMOVD X1, AX
	VCVTPH2PS X1, X1
	VADDSS X1, X0, X0

	ADDQ $2, SI
	DECQ CX
	JNZ  sum_fp16_scalar

sum_fp16_store:
	FP16_NARROW_SCALAR_F32_X0_TO(ret+16(FP))
	RET

sum_fp16_zero:
	XORPS X0, X0
	FP16_NARROW_SCALAR_F32_X0_TO(ret+16(FP))
	RET
