#include "textflag.h"

#define VCVTPS2PH_Y0_X2 WORD $0xC4E3; WORD $0x7D1D; BYTE $0xD8; BYTE $0x00

#define WIDEN_FP16_4H(srcReg, dstY) \
	VMOVDQU X2, (srcReg); \
	VCVTPH2PS X2, dstY

#define NARROW_FP16_Y0_TO_4H(dstReg) \
	VCVTPS2PH_Y0_X2; \
	VMOVDQU X2, (dstReg)

// func ConvTranspose2dTapFP16AVX512Asm(outRow *uint16, weightVal float32, inputCol *uint16, outCols int)
TEXT ·ConvTranspose2dTapFP16AVX512Asm(SB), NOSPLIT, $0-32
	MOVQ outRow+0(FP), DI
	MOVSS weightVal+8(FP), X1
	VBROADCASTSS X1, Y2
	MOVQ inputCol+16(FP), SI
	MOVQ outCols+24(FP), CX

	CMPQ CX, $4
	JL   ct_fp16_tap_done

	WIDEN_FP16_4H(DI, Y0)
	WIDEN_FP16_4H(SI, Y1)
	VFMADD231PS Y0, Y1, Y2
	NARROW_FP16_Y0_TO_4H(DI)

ct_fp16_tap_done:
	RET
