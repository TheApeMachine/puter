#include "textflag.h"
#include "../avx512_fp16_macros.inc"

#define HORIZ_MAX_FP16_X0 \
	VPSRLDQ $8, X0, X1; \
	VMAXPH_X0_X1_X0; \
	VPSRLDQ $4, X0, X1; \
	VMAXPH_X0_X1_X0; \
	VPSRLDQ $2, X0, X1; \
	VMAXPH_X0_X1_X0

// func MaxFloat16AVX512Asm(src *uint16, count int) float32
TEXT ·MaxFloat16AVX512Asm(SB), NOSPLIT, $16-20
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX

	TESTQ CX, CX
	JZ    max_fp16_avx512_zero

	MOVWLZX (SI), AX
	MOVW AX, 0(SP)
	VPBROADCASTW 0(SP), X14

	ADDQ $2, SI
	DECQ CX

max_fp16_avx512_w8:
	CMPQ CX, $8
	JL    max_fp16_avx512_w4

	VMOVUPH_Y1_SI
	VEXTRACTI128 $0, Y1, X0
	HORIZ_MAX_FP16_X0
	VMAXPH_X14_X0_X14

	ADDQ $16, SI
	SUBQ $8, CX
	JMP  max_fp16_avx512_w8

max_fp16_avx512_w4:
	CMPQ CX, $4
	JL    max_fp16_avx512_tail

	VMOVDQU X0, (SI)
	HORIZ_MAX_FP16_X0
	VMAXPH_X14_X0_X14

	ADDQ $8, SI
	SUBQ $4, CX
	JMP  max_fp16_avx512_w4

max_fp16_avx512_tail:
	TESTQ CX, CX
	JZ    max_fp16_avx512_store

max_fp16_avx512_scalar:
	MOVWLZX (SI), AX
	MOVW AX, 8(SP)
	VPBROADCASTW 8(SP), X0
	VMAXPH_X14_X0_X14

	ADDQ $2, SI
	DECQ CX
	JNZ  max_fp16_avx512_scalar

max_fp16_avx512_store:
	VCVTPH2PS X14, X0
	MOVSS X0, ret+16(FP)
	RET

max_fp16_avx512_zero:
	XORPS X0, X0
	MOVSS X0, ret+16(FP)
	RET
