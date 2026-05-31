#include "textflag.h"
#include "../avx512_bf16_macros.inc"

// func SumBFloat16AVX512Asm(src *uint16, count int) uint16
TEXT ·SumBFloat16AVX512Asm(SB), NOSPLIT, $0-18
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX

	TESTQ CX, CX
	JZ    sum_bf16_zero

	VXORPS Y0, Y0, Y0

sum_bf16_w8:
	CMPQ CX, $8
	JL   sum_bf16_w4

	BF16_LOAD_8H(SI, Y1)
	VADDPS Y1, Y0, Y0

	ADDQ $16, SI
	SUBQ $8, CX
	JMP  sum_bf16_w8

sum_bf16_w4:
	CMPQ CX, $4
	JL   sum_bf16_reduce

	BF16_LOAD_4H(SI, Y1)
	VADDPS Y1, Y0, Y0

	ADDQ $8, SI
	SUBQ $4, CX
	JMP  sum_bf16_w4

sum_bf16_reduce:
	VEXTRACTF128 $1, Y0, X1
	VADDPS       X1, X0, X0
	VHADDPS      X0, X0, X0
	VHADDPS      X0, X0, X0

	TESTQ CX, CX
	JZ    sum_bf16_store

sum_bf16_scalar:
	MOVWLZX (SI), AX
	SHLQ  $16, AX
	VMOVD X1, AX
	VADDSS X1, X0, X0

	ADDQ $2, SI
	DECQ CX
	JNZ  sum_bf16_scalar

sum_bf16_store:
	PACK_BF16_SCALAR_F32_X0_TO(ret+16(FP))
	RET

sum_bf16_zero:
	XORPS X0, X0
	PACK_BF16_SCALAR_F32_X0_TO(ret+16(FP))
	RET
