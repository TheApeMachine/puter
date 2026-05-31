#include "textflag.h"
#include "../avx512_bf16_macros.inc"
#include "../f16c_fp16_macros.inc"

DATA prodOneF32AVX2<>+0(SB)/4, $0x3f800000
GLOBL prodOneF32AVX2<>(SB), RODATA|NOPTR, $4

// func ProdBFloat16AVX2Asm(src *uint16, count int) float32
TEXT ·ProdBFloat16AVX2Asm(SB), NOSPLIT, $0-20
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX

	TESTQ CX, CX
	JZ    prod_bf16_avx2_zero

	VBROADCASTSS prodOneF32AVX2<>(SB), Y0

prod_bf16_avx2_w8:
	CMPQ CX, $8
	JL    prod_bf16_avx2_w4

	BF16_LOAD_8H(SI, Y1)
	VMULPS Y1, Y0, Y0

	ADDQ $16, SI
	SUBQ $8, CX
	JMP  prod_bf16_avx2_w8

prod_bf16_avx2_w4:
	CMPQ CX, $4
	JL    prod_bf16_avx2_tail

	BF16_LOAD_4H(SI, Y1)
	VMULPS Y1, Y0, Y0

	ADDQ $8, SI
	SUBQ $4, CX
	JMP  prod_bf16_avx2_w4

prod_bf16_avx2_tail:
	VEXTRACTF128 $1, Y0, X1
	VMULPS       X1, X0, X0
	VHADDPS      X0, X0, X0
	VHADDPS      X0, X0, X0

	TESTQ CX, CX
	JZ    prod_bf16_avx2_store

prod_bf16_avx2_scalar:
	MOVWLZX (SI), AX
	SHLQ  $16, AX
	VMOVD X1, AX
	VMULSS X1, X0, X0

	ADDQ $2, SI
	DECQ CX
	JNZ  prod_bf16_avx2_scalar

prod_bf16_avx2_store:
	VEXTRACTF128 $0, Y0, X0
	MOVSS X0, ret+16(FP)
	RET

prod_bf16_avx2_zero:
	XORPS X0, X0
	MOVSS X0, ret+16(FP)
	RET

// func ProdFloat16AVX2Asm(src *uint16, count int) float32
TEXT ·ProdFloat16AVX2Asm(SB), NOSPLIT, $0-20
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX

	TESTQ CX, CX
	JZ    prod_fp16_avx2_zero

	VBROADCASTSS prodOneF32AVX2<>(SB), Y0

prod_fp16_avx2_w8:
	CMPQ CX, $8
	JL    prod_fp16_avx2_w4

	FP16_WIDEN_AVX2_8H(SI, Y1)
	VMULPS Y1, Y0, Y0

	ADDQ $16, SI
	SUBQ $8, CX
	JMP  prod_fp16_avx2_w8

prod_fp16_avx2_w4:
	CMPQ CX, $4
	JL    prod_fp16_avx2_tail

	FP16_WIDEN_AVX2_4H(SI, Y1)
	VMULPS Y1, Y0, Y0

	ADDQ $8, SI
	SUBQ $4, CX
	JMP  prod_fp16_avx2_w4

prod_fp16_avx2_tail:
	VEXTRACTF128 $1, Y0, X1
	VMULPS       X1, X0, X0
	VHADDPS      X0, X0, X0
	VHADDPS      X0, X0, X0

	TESTQ CX, CX
	JZ    prod_fp16_avx2_store

prod_fp16_avx2_scalar:
	MOVWLZX (SI), AX
	VMOVD X1, AX
	VCVTPH2PS X1, X1
	VMULSS X1, X0, X0

	ADDQ $2, SI
	DECQ CX
	JNZ  prod_fp16_avx2_scalar

prod_fp16_avx2_store:
	VEXTRACTF128 $0, Y0, X0
	MOVSS X0, ret+16(FP)
	RET

prod_fp16_avx2_zero:
	XORPS X0, X0
	MOVSS X0, ret+16(FP)
	RET
