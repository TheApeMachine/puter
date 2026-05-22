#include "textflag.h"

DATA l1FP16AbsMaskAVX512<>+0(SB)/4, $0x7fffffff
DATA l1FP16AbsMaskAVX512<>+4(SB)/4, $0x7fffffff
DATA l1FP16AbsMaskAVX512<>+8(SB)/4, $0x7fffffff
DATA l1FP16AbsMaskAVX512<>+12(SB)/4, $0x7fffffff
GLOBL l1FP16AbsMaskAVX512<>(SB), RODATA|NOPTR, $16

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

// func L1NormFloat16AVX512Asm(src *uint16, count int) float32
TEXT ·L1NormFloat16AVX512Asm(SB), NOSPLIT, $0-20
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX

	TESTQ CX, CX
	JZ    l1_fp16_avx512_zero

	VXORPS Y0, Y0, Y0
	VMOVUPS l1FP16AbsMaskAVX512<>(SB), X6
	VINSERTF128 $1, X6, Y6, Y6

l1_fp16_avx512_w8:
	CMPQ CX, $8
	JL    l1_fp16_avx512_w4

	WIDEN_FP16_8H(SI, Y1)
	VANDPS Y6, Y1, Y1
	VADDPS Y1, Y0, Y0

	ADDQ $16, SI
	SUBQ $8, CX
	JMP  l1_fp16_avx512_w8

l1_fp16_avx512_w4:
	CMPQ CX, $4
	JL    l1_fp16_avx512_reduce

	WIDEN_FP16_4H(SI, Y1)
	VANDPS Y6, Y1, Y1
	VADDPS Y1, Y0, Y0

	ADDQ $8, SI
	SUBQ $4, CX
	JMP  l1_fp16_avx512_w4

l1_fp16_avx512_reduce:
	VEXTRACTF128 $1, Y0, X1
	VADDPS       X1, X0, X0
	VHADDPS      X0, X0, X0
	VHADDPS      X0, X0, X0

	TESTQ CX, CX
	JZ    l1_fp16_avx512_store

l1_fp16_avx512_scalar:
	MOVWLZX (SI), AX
	VMOVD X1, AX
	VCVTPH2PS X1, X1
	VANDPS X6, X1, X1
	VADDSS X1, X0, X0

	ADDQ $2, SI
	DECQ CX
	JNZ  l1_fp16_avx512_scalar

l1_fp16_avx512_store:
	MOVSS X0, ret+16(FP)
	RET

l1_fp16_avx512_zero:
	XORPS X0, X0
	MOVSS X0, ret+16(FP)
	RET
