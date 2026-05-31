#include "textflag.h"
#include "../avx512_bf16_macros.inc"

DATA l1BF16AbsMaskAVX512<>+0(SB)/4, $0x7fffffff
DATA l1BF16AbsMaskAVX512<>+4(SB)/4, $0x7fffffff
DATA l1BF16AbsMaskAVX512<>+8(SB)/4, $0x7fffffff
DATA l1BF16AbsMaskAVX512<>+12(SB)/4, $0x7fffffff
GLOBL l1BF16AbsMaskAVX512<>(SB), RODATA|NOPTR, $16

// func L1NormBFloat16AVX512Asm(src *uint16, count int) float32
TEXT ·L1NormBFloat16AVX512Asm(SB), NOSPLIT, $0-20
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX

	TESTQ CX, CX
	JZ    l1_bf16_avx512_zero

	VXORPS Y0, Y0, Y0
	VMOVUPS l1BF16AbsMaskAVX512<>(SB), X6
	VINSERTF128 $1, X6, Y6, Y6

l1_bf16_avx512_w8:
	CMPQ CX, $8
	JL    l1_bf16_avx512_w4

	BF16_LOAD_8H(SI, Y1)
	VANDPS Y6, Y1, Y1
	VADDPS Y1, Y0, Y0

	ADDQ $16, SI
	SUBQ $8, CX
	JMP  l1_bf16_avx512_w8

l1_bf16_avx512_w4:
	CMPQ CX, $4
	JL    l1_bf16_avx512_reduce

	BF16_LOAD_4H(SI, Y1)
	VANDPS Y6, Y1, Y1
	VADDPS Y1, Y0, Y0

	ADDQ $8, SI
	SUBQ $4, CX
	JMP  l1_bf16_avx512_w4

l1_bf16_avx512_reduce:
	VEXTRACTF128 $1, Y0, X1
	VADDPS       X1, X0, X0
	VHADDPS      X0, X0, X0
	VHADDPS      X0, X0, X0

	TESTQ CX, CX
	JZ    l1_bf16_avx512_store

l1_bf16_avx512_scalar:
	MOVWLZX (SI), AX
	SHLQ  $16, AX
	VMOVD X1, AX
	VANDPS X6, X1, X1
	VADDSS X1, X0, X0

	ADDQ $2, SI
	DECQ CX
	JNZ  l1_bf16_avx512_scalar

l1_bf16_avx512_store:
	MOVSS X0, ret+16(FP)
	RET

l1_bf16_avx512_zero:
	XORPS X0, X0
	MOVSS X0, ret+16(FP)
	RET
