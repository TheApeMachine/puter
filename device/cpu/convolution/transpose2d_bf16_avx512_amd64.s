#include "textflag.h"
#include "../avx512_bf16_macros.inc"

// func ConvTranspose2dTapBF16AVX512Asm(outRow *uint16, weightVal float32, inputCol *uint16, outCols int)
TEXT ·ConvTranspose2dTapBF16AVX512Asm(SB), NOSPLIT, $0-32
	MOVQ outRow+0(FP), DI
	MOVSS weightVal+8(FP), X1
	VBROADCASTSS X1, Y2
	MOVQ inputCol+16(FP), SI
	MOVQ outCols+24(FP), CX

	CMPQ CX, $4
	JL   ct_bf16_tap_done

	BF16_LOAD_4H(DI, Y0)
	BF16_LOAD_4H(SI, Y1)
	VFMADD231PS Y0, Y1, Y2
	PACK_BF16_4H(DI)

ct_bf16_tap_done:
	RET
