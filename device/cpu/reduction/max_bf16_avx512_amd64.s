#include "textflag.h"

#define WIDEN_BF16_8H(baseReg, dstY) \
	VMOVDQU X1, (baseReg); \
	VPMOVZXWD X1, dstY; \
	VPSLLD $16, dstY, dstY; \
	VPSRLDQ $8, X1, X2; \
	VPMOVZXWD X2, Y3; \
	VPSLLD $16, Y3, Y3; \
	VEXTRACTI128 $0, Y3, X3; \
	VINSERTF128 $1, X3, dstY, dstY

#define WIDEN_BF16_4H(baseReg, dstY) \
	VMOVDQU X1, (baseReg); \
	VPMOVZXWD X1, dstY; \
	VPSLLD $16, dstY, dstY

// func MaxBFloat16AVX512Asm(src *uint16, count int) float32
TEXT ·MaxBFloat16AVX512Asm(SB), NOSPLIT, $0-20
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX

	TESTQ CX, CX
	JZ    max_bf16_avx512_zero

	MOVWLZX (SI), AX
	SHLQ  $16, AX
	VMOVD X0, AX
	VBROADCASTSS X0, Y0

	ADDQ $2, SI
	DECQ CX

max_bf16_avx512_w8:
	CMPQ CX, $8
	JL    max_bf16_avx512_w4

	WIDEN_BF16_8H(SI, Y1)
	VMAXPS Y1, Y0, Y0

	ADDQ $16, SI
	SUBQ $8, CX
	JMP  max_bf16_avx512_w8

max_bf16_avx512_w4:
	CMPQ CX, $4
	JL    max_bf16_avx512_reduce

	WIDEN_BF16_4H(SI, Y1)
	VMAXPS Y1, Y0, Y0

	ADDQ $8, SI
	SUBQ $4, CX
	JMP  max_bf16_avx512_w4

max_bf16_avx512_reduce:
	VEXTRACTF128 $1, Y0, X1
	VMAXPS       X1, X0, X0
	VHADDPS      X0, X0, X0
	VHADDPS      X0, X0, X0

	TESTQ CX, CX
	JZ    max_bf16_avx512_store

max_bf16_avx512_scalar:
	MOVWLZX (SI), AX
	SHLQ  $16, AX
	VMOVD X1, AX
	VMAXSS X1, X0, X0

	ADDQ $2, SI
	DECQ CX
	JNZ  max_bf16_avx512_scalar

max_bf16_avx512_store:
	MOVSS X0, ret+16(FP)
	RET

max_bf16_avx512_zero:
	XORPS X0, X0
	MOVSS X0, ret+16(FP)
	RET
