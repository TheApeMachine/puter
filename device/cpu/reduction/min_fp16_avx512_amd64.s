#include "textflag.h"
#include "../avx512_fp16_macros.inc"

DATA minFP16PosHalfAVX512<>+0(SB)/2, $0x7c00
GLOBL minFP16PosHalfAVX512<>(SB), RODATA|NOPTR, $2

#define HORIZ_MIN_FP16_X0 \
	VPSRLDQ $8, X0, X1; \
	VMINPH_X0_X1_X0; \
	VPSRLDQ $4, X0, X1; \
	VMINPH_X0_X1_X0; \
	VPSRLDQ $2, X0, X1; \
	VMINPH_X0_X1_X0

// func MinFloat16AVX512Asm(src *uint16, count int) float32
TEXT ·MinFloat16AVX512Asm(SB), NOSPLIT, $16-20
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX

	TESTQ CX, CX
	JZ    min_fp16_avx512_zero

	VPBROADCASTW minFP16PosHalfAVX512<>(SB), X14

min_fp16_avx512_w8:
	CMPQ CX, $8
	JL    min_fp16_avx512_w4

	VMOVUPH_Y1_SI
	VEXTRACTI128 $0, Y1, X0
	HORIZ_MIN_FP16_X0
	VMINPH_X14_X0_X14

	ADDQ $16, SI
	SUBQ $8, CX
	JMP  min_fp16_avx512_w8

min_fp16_avx512_w4:
	CMPQ CX, $4
	JL    min_fp16_avx512_tail

	VMOVDQU X0, (SI)
	HORIZ_MIN_FP16_X0
	VMINPH_X14_X0_X14

	ADDQ $8, SI
	SUBQ $4, CX
	JMP  min_fp16_avx512_w4

min_fp16_avx512_tail:
	TESTQ CX, CX
	JZ    min_fp16_avx512_store

min_fp16_avx512_scalar:
	MOVWLZX (SI), AX
	MOVW AX, 0(SP)
	VPBROADCASTW 0(SP), X0
	VMINPH_X14_X0_X14

	ADDQ $2, SI
	DECQ CX
	JNZ  min_fp16_avx512_scalar

min_fp16_avx512_store:
	VCVTPH2PS X14, X0
	MOVSS X0, ret+16(FP)
	RET

min_fp16_avx512_zero:
	XORPS X0, X0
	MOVSS X0, ret+16(FP)
	RET
