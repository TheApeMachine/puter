#include "textflag.h"
#include "../avx512_fp16_macros.inc"

DATA prodFP16OneHalfAVX512<>+0(SB)/2, $0x3c00
GLOBL prodFP16OneHalfAVX512<>(SB), RODATA|NOPTR, $2

#define HORIZ_MUL_FP16_X0 \
	VPSRLDQ $8, X0, X1; \
	VMULPH_X0_X1_X0; \
	VPSRLDQ $4, X0, X1; \
	VMULPH_X0_X1_X0; \
	VPSRLDQ $2, X0, X1; \
	VMULPH_X0_X1_X0

#define HORIZ_MAX_FP16_X0 \
	VPSRLDQ $8, X0, X1; \
	VMAXPH_X0_X1_X0; \
	VPSRLDQ $4, X0, X1; \
	VMAXPH_X0_X1_X0; \
	VPSRLDQ $2, X0, X1; \
	VMAXPH_X0_X1_X0

#define HORIZ_MIN_FP16_X0 \
	VPSRLDQ $8, X0, X1; \
	VMINPH_X0_X1_X0; \
	VPSRLDQ $4, X0, X1; \
	VMINPH_X0_X1_X0; \
	VPSRLDQ $2, X0, X1; \
	VMINPH_X0_X1_X0

// func ProdFloat16AVX512Asm(src *uint16, count int) float32
TEXT ·ProdFloat16AVX512Asm(SB), NOSPLIT, $0-20
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX

	TESTQ CX, CX
	JZ    prod_fp16_avx512_zero

	MOVSS prodOneF32<>(SB), X15

prod_fp16_avx512_w8:
	CMPQ CX, $8
	JL    prod_fp16_avx512_w4

	VMOVUPH_Y0_SI
	VEXTRACTI128 $0, Y0, X0
	HORIZ_MUL_FP16_X0
	VCVTPH2PS X0, X1
	VMULSS X1, X15, X15

	ADDQ $16, SI
	SUBQ $8, CX
	JMP  prod_fp16_avx512_w8

prod_fp16_avx512_w4:
	CMPQ CX, $4
	JL    prod_fp16_avx512_tail

	VMOVDQU X0, (SI)
	HORIZ_MUL_FP16_X0
	VCVTPH2PS X0, X1
	VMULSS X1, X15, X15

	ADDQ $8, SI
	SUBQ $4, CX
	JMP  prod_fp16_avx512_w4

prod_fp16_avx512_tail:
	TESTQ CX, CX
	JZ    prod_fp16_avx512_store

prod_fp16_avx512_scalar:
	MOVWLZX (SI), AX
	MOVW AX, 0(SP)
	VPBROADCASTW 0(SP), X0
	VCVTPH2PS X0, X1
	VMULSS X1, X15, X15

	ADDQ $2, SI
	DECQ CX
	JNZ  prod_fp16_avx512_scalar

prod_fp16_avx512_store:
	MOVSS X15, ret+16(FP)
	RET

prod_fp16_avx512_zero:
	XORPS X15, X15
	MOVSS X15, ret+16(FP)
	RET

DATA prodOneF32<>(SB)/4, $0x3f800000
GLOBL prodOneF32<>(SB), RODATA|NOPTR, $4
