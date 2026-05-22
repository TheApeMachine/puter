#include "textflag.h"

#define WIDEN_BF16_4H(srcReg, dstY) \
	VMOVDQU X2, (srcReg); \
	VPMOVZXWD X2, dstY; \
	VPSLLD $16, dstY, dstY

#define NARROW_BF16_Y0_TO_4H(dstReg) \
	VPSRLD $16, Y0, Y0; \
	VEXTRACTI128 $0, Y0, X2; \
	MOVL  X2, AX; \
	MOVW  AX, (dstReg); \
	PEXTRD $1, X2, AX; \
	MOVW  AX, 2(dstReg); \
	PEXTRD $2, X2, AX; \
	MOVW  AX, 4(dstReg); \
	PEXTRD $3, X2, AX; \
	MOVW  AX, 6(dstReg)

// func ConvTranspose2dTapBF16AVX512Asm(outRow *uint16, weightVal float32, inputCol *uint16, outCols int)
TEXT ·ConvTranspose2dTapBF16AVX512Asm(SB), NOSPLIT, $0-32
	MOVQ outRow+0(FP), DI
	MOVSS weightVal+8(FP), X1
	VBROADCASTSS X1, Y2
	MOVQ inputCol+16(FP), SI
	MOVQ outCols+24(FP), CX

	CMPQ CX, $4
	JL   ct_bf16_tap_done

	WIDEN_BF16_4H(DI, Y0)
	WIDEN_BF16_4H(SI, Y1)
	VFMADD231PS Y0, Y1, Y2
	NARROW_BF16_Y0_TO_4H(DI)

ct_bf16_tap_done:
	RET
