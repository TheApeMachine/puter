#include "textflag.h"
#include "../avx512_bf16_macros.inc"

// func DotBFloat16AVX512Asm(left, right *uint16, count int) uint16
TEXT ·DotBFloat16AVX512Asm(SB), NOSPLIT, $0-26
	MOVQ left+0(FP), SI
	MOVQ right+8(FP), DI
	MOVQ count+16(FP), CX

	TESTQ CX, CX
	JZ    dot_bf16_zero

	VXORPS Y0, Y0, Y0

dot_bf16_w8:
	CMPQ CX, $8
	JL   dot_bf16_w4

	BF16_LOAD_8H(SI, Y1)
	BF16_LOAD_8H(DI, Y2)
	VMULPS Y2, Y1, Y1
	VADDPS Y1, Y0, Y0

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $8, CX
	JMP  dot_bf16_w8

dot_bf16_w4:
	CMPQ CX, $4
	JL   dot_bf16_reduce

	BF16_LOAD_4H(SI, Y1)
	BF16_LOAD_4H(DI, Y2)
	VMULPS Y2, Y1, Y1
	VADDPS Y1, Y0, Y0

	ADDQ $8, SI
	ADDQ $8, DI
	SUBQ $4, CX
	JMP  dot_bf16_w4

dot_bf16_reduce:
	VEXTRACTF128 $1, Y0, X1
	VADDPS       X1, X0, X0
	VHADDPS      X0, X0, X0
	VHADDPS      X0, X0, X0

	TESTQ CX, CX
	JZ    dot_bf16_store

dot_bf16_tail:
	MOVWLZX (SI), AX
	SHLQ  $16, AX
	VMOVD X1, AX
	MOVWLZX (DI), DX
	SHLQ  $16, DX
	VMOVD X2, DX
	VMULSS X2, X1, X1
	VADDSS X1, X0, X0

	ADDQ $2, SI
	ADDQ $2, DI
	DECQ CX
	JNZ  dot_bf16_tail

dot_bf16_store:
	PACK_BF16_SCALAR_F32_X0_TO(ret+24(FP))
	RET

dot_bf16_zero:
	XORPS X0, X0
	PACK_BF16_SCALAR_F32_X0_TO(ret+24(FP))
	RET
