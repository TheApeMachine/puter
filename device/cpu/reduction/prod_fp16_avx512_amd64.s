#include "textflag.h"

DATA prodFP16OneF32AVX512<>+0(SB)/4, $0x3f800000
GLOBL prodFP16OneF32AVX512<>(SB), RODATA|NOPTR, $4

#define VCVTPS2PH_X0_X2 WORD $0xC4E3; WORD $0x7D1D; BYTE $0xD0; BYTE $0x00

#define WIDEN_FP16_8H(baseReg, dstY) \
	VMOVDQU X1, (baseReg); \
	VCVTPH2PS X1, dstY; \
	VPSRLDQ $8, X1, X2; \
	VCVTPH2PS X2, Y3; \
	VEXTRACTI128 $0, Y3, X3; \
	VINSERTF128 $1, X3, dstY, dstY

#define WIDEN_FP16_4H(baseReg, dstY) \
	VMOVDQU X1, (baseReg); \
	VCVTPH2PS X1, dstY

// func ProdFloat16AVX512Asm(src *uint16, count int) float32
TEXT ·ProdFloat16AVX512Asm(SB), NOSPLIT, $0-20
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX

	TESTQ CX, CX
	JZ    prod_fp16_avx512_zero

	VBROADCASTSS prodFP16OneF32AVX512<>(SB), Y0

prod_fp16_avx512_w8:
	CMPQ CX, $8
	JL    prod_fp16_avx512_w4

	WIDEN_FP16_8H(SI, Y1)
	VMULPS Y1, Y0, Y0

	ADDQ $16, SI
	SUBQ $8, CX
	JMP  prod_fp16_avx512_w8

prod_fp16_avx512_w4:
	CMPQ CX, $4
	JL    prod_fp16_avx512_reduce

	WIDEN_FP16_4H(SI, Y1)
	VMULPS Y1, Y0, Y0

	ADDQ $8, SI
	SUBQ $4, CX
	JMP  prod_fp16_avx512_w4

prod_fp16_avx512_reduce:
	VEXTRACTF128 $1, Y0, X1
	VMULPS       X1, X0, X0
	VHADDPS      X0, X0, X0
	VHADDPS      X0, X0, X0

	TESTQ CX, CX
	JZ    prod_fp16_avx512_store

prod_fp16_avx512_scalar:
	MOVWLZX (SI), AX
	VMOVD X1, AX
	VCVTPH2PS X1, X1
	VMULSS X1, X0, X0

	ADDQ $2, SI
	DECQ CX
	JNZ  prod_fp16_avx512_scalar

prod_fp16_avx512_store:
	MOVSS X0, ret+16(FP)
	RET

prod_fp16_avx512_zero:
	XORPS X0, X0
	MOVSS X0, ret+16(FP)
	RET
