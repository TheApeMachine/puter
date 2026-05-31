#include "textflag.h"
#include "../avx512_fp16_macros.inc"
#include "../f16c_fp16_macros.inc"

#define ACCUM_FP16_PRODUCTS_Y3 \
	VEXTRACTI128 $0, Y3, X10; \
	VEXTRACTI128 $1, Y3, X11; \
	VCVTPH2PS X10, Y12; \
	VCVTPH2PS X11, Y13; \
	VADDPS Y12, Y0, Y0; \
	VADDPS Y13, Y0, Y0

#define ACCUM_FP16_PRODUCTS_X3 \
	VCVTPH2PS X3, X12; \
	VADDPS X12, X0, X0

// func MatmulRowFP16AVX512Asm(cRow, aRow, b *uint16, inner, colsBlock, bCols int)
TEXT ·MatmulRowFP16AVX512Asm(SB), NOSPLIT, $16-48
	MOVQ cRow+0(FP), DI
	MOVQ aRow+8(FP), SI
	MOVQ b+16(FP), BX
	MOVQ inner+24(FP), R12
	MOVQ colsBlock+32(FP), DX
	MOVQ bCols+40(FP), R15

	MOVQ R15, R8
	SHLQ $1, R8

mm_col_loop:
	CMPQ DX, $8
	JL   mm_col_w4

	VXORPS Y0, Y0, Y0

	MOVQ R12, CX
	MOVQ SI, R13
	MOVQ BX, R14

mm_k8:
	TESTQ CX, CX
	JZ    mm_k8_done

	MOVWLZX (R13), AX
	MOVW AX, 0(SP)
	VPBROADCASTW 0(SP), Y1

	VMOVDQU Y2, (R14)
	VMULPH_Y3_Y1_Y2
	ACCUM_FP16_PRODUCTS_Y3

	ADDQ $2, R13
	ADDQ R8, R14
	DECQ CX
	JMP  mm_k8

mm_k8_done:
	VCVTPS2PH_Y0_X2
	STORE_Y0_16H_DI

	ADDQ $16, DI
	ADDQ $16, BX
	SUBQ $8, DX
	JMP  mm_col_loop

mm_col_w4:
	CMPQ DX, $4
	JL   mm_done

	XORPS X0, X0

	MOVQ R12, CX
	MOVQ SI, R13
	MOVQ BX, R14

mm_k4:
	TESTQ CX, CX
	JZ    mm_k4_done

	MOVWLZX (R13), AX
	MOVW AX, 0(SP)
	VPBROADCASTW 0(SP), X1

	VMOVDQU X2, (R14)
	VMULPH_X3_X0_X1
	ACCUM_FP16_PRODUCTS_X3

	ADDQ $2, R13
	ADDQ R8, R14
	DECQ CX
	JMP  mm_k4

mm_k4_done:
	VCVTPS2PH_X0_X2
	STORE_X0_8H_DI

	ADDQ $8, DI
	ADDQ $8, BX
	SUBQ $4, DX
	JMP  mm_col_loop

mm_done:
	RET
