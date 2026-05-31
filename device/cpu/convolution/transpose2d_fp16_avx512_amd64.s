#include "textflag.h"
#include "../avx512_fp16_macros.inc"
#include "../f16c_fp16_macros.inc"

// func ConvTranspose2dTapFP16AVX512Asm(outRow *uint16, weightVal float32, inputCol *uint16, outCols int)
TEXT ·ConvTranspose2dTapFP16AVX512Asm(SB), NOSPLIT, $16-32
	MOVQ outRow+0(FP), DI
	MOVSS weightVal+8(FP), X1
	MOVQ inputCol+16(FP), SI
	MOVQ outCols+24(FP), CX

	CMPQ CX, $4
	JL   ct_fp16_tap_done

	VMOVAPS X1, X0
	VCVTPS2PH_X0_X2
	VPBROADCASTW X2, X2

	VMOVDQU X0, (DI)
	VMOVDQU X3, (SI)
	VMULPH_X4_X3_X2
	VADDPH_X0_X4_X0
	STORE_X0_8H_DI

ct_fp16_tap_done:
	RET
