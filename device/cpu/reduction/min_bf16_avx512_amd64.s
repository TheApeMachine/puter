#include "textflag.h"
#include "../avx512_bf16_macros.inc"

// func MinBFloat16AVX512Asm(src *uint16, count int) float32
TEXT ·MinBFloat16AVX512Asm(SB), NOSPLIT, $0-20
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX

	TESTQ CX, CX
	JZ    min_bf16_avx512_zero

	MOVWLZX (SI), AX
	SHLQ  $16, AX
	VMOVD X0, AX
	VBROADCASTSS X0, Y0

	ADDQ $2, SI
	DECQ CX

min_bf16_avx512_w8:
	CMPQ CX, $8
	JL    min_bf16_avx512_w4

	BF16_LOAD_8H(SI, Y1)
	VMINPS Y1, Y0, Y0

	ADDQ $16, SI
	SUBQ $8, CX
	JMP  min_bf16_avx512_w8

min_bf16_avx512_w4:
	CMPQ CX, $4
	JL    min_bf16_avx512_reduce

	BF16_LOAD_4H(SI, Y1)
	VMINPS Y1, Y0, Y0

	ADDQ $8, SI
	SUBQ $4, CX
	JMP  min_bf16_avx512_w4

min_bf16_avx512_reduce:
	VEXTRACTF128 $1, Y0, X1
	VMINPS       X1, X0, X0
	VHADDPS      X0, X0, X0
	VHADDPS      X0, X0, X0

	TESTQ CX, CX
	JZ    min_bf16_avx512_store

min_bf16_avx512_scalar:
	MOVWLZX (SI), AX
	SHLQ  $16, AX
	VMOVD X1, AX
	VMINSS X1, X0, X0

	ADDQ $2, SI
	DECQ CX
	JNZ  min_bf16_avx512_scalar

min_bf16_avx512_store:
	MOVSS X0, ret+16(FP)
	RET

min_bf16_avx512_zero:
	XORPS X0, X0
	MOVSS X0, ret+16(FP)
	RET
