#include "textflag.h"

DATA l1ReducedAbsMaskAVX2<>+0(SB)/4, $0x7fffffff
DATA l1ReducedAbsMaskAVX2<>+4(SB)/4, $0x7fffffff
DATA l1ReducedAbsMaskAVX2<>+8(SB)/4, $0x7fffffff
DATA l1ReducedAbsMaskAVX2<>+12(SB)/4, $0x7fffffff
GLOBL l1ReducedAbsMaskAVX2<>(SB), RODATA|NOPTR, $16

#define WIDEN_BF16_8H_AVX2(baseReg, dstY) \
	VMOVDQU X1, (baseReg); \
	VPXOR X3, X3, X3; \
	VPUNPCKLWD X3, X1, X4; \
	VPUNPCKHWD X3, X1, X5; \
	VPSLLD $16, X4, X4; \
	VPSLLD $16, X5, X5; \
	VINSERTF128 $1, X5, Y4, dstY

#define WIDEN_BF16_4H_AVX2(baseReg, dstY) \
	VMOVDQU X1, (baseReg); \
	VPXOR X3, X3, X3; \
	VPUNPCKLWD X3, X1, X4; \
	VPUNPCKHWD X3, X1, X5; \
	VPSLLD $16, X4, X4; \
	VPSLLD $16, X5, X5; \
	VUNPCKLPD X4, X5, dstY

#define WIDEN_FP16_8H_AVX2(baseReg, dstY) \
	VMOVDQU X1, (baseReg); \
	VCVTPH2PS X1, dstY; \
	VPSRLDQ $8, X1, X2; \
	VCVTPH2PS X2, X5; \
	VINSERTF128 $1, X5, dstY, dstY

#define BF16_MINMAX_REDUCE_AVX2 \
	VEXTRACTF128 $1, Y0, X1; \
	VMINPS       X1, X0, X0; \
	VHADDPS      X0, X0, X0; \
	VHADDPS      X0, X0, X0

#define BF16_MAX_REDUCE_AVX2 \
	VEXTRACTF128 $1, Y0, X1; \
	VMAXPS       X1, X0, X0; \
	VHADDPS      X0, X0, X0; \
	VHADDPS      X0, X0, X0

#define BF16_L1_REDUCE_AVX2 \
	VEXTRACTF128 $1, Y0, X1; \
	VADDPS       X1, X0, X0; \
	VHADDPS      X0, X0, X0; \
	VHADDPS      X0, X0, X0

#define STORE_XMM0_F32 \
	MOVSS X0, ret+16(FP); \
	RET

// func MinBFloat16AVX2Asm(src *uint16, count int) float32
TEXT ·MinBFloat16AVX2Asm(SB), NOSPLIT, $0-20
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX

	TESTQ CX, CX
	JZ    min_bf16_avx2_zero

	MOVWLZX (SI), AX
	SHLQ  $16, AX
	VMOVD X0, AX
	VBROADCASTSS X0, Y0

	ADDQ $2, SI
	DECQ CX

min_bf16_avx2_w8:
	CMPQ CX, $8
	JL    min_bf16_avx2_w4

	WIDEN_BF16_8H_AVX2(SI, Y1)
	VMINPS Y1, Y0, Y0

	ADDQ $16, SI
	SUBQ $8, CX
	JMP  min_bf16_avx2_w8

min_bf16_avx2_w4:
	CMPQ CX, $4
	JL    min_bf16_avx2_tail

	VMOVDQU X1, (SI)
	VPXOR X3, X3, X3
	VPUNPCKLWD X3, X1, X4
	VPUNPCKHWD X3, X1, X5
	VPSLLD $16, X4, X4
	VPSLLD $16, X5, X5
	VINSERTF128 $1, X5, Y4, Y1
	VMINPS Y1, Y0, Y0

	ADDQ $8, SI
	SUBQ $4, CX
	JMP  min_bf16_avx2_w4

min_bf16_avx2_tail:
	BF16_MINMAX_REDUCE_AVX2

	TESTQ CX, CX
	JZ    min_bf16_avx2_store

min_bf16_avx2_scalar:
	MOVWLZX (SI), AX
	SHLQ  $16, AX
	VMOVD X1, AX
	VMINSS X1, X0, X0

	ADDQ $2, SI
	DECQ CX
	JNZ  min_bf16_avx2_scalar

min_bf16_avx2_store:
	STORE_XMM0_F32

min_bf16_avx2_zero:
	XORPS X0, X0
	MOVSS X0, ret+16(FP)
	RET

// func MaxBFloat16AVX2Asm(src *uint16, count int) float32
TEXT ·MaxBFloat16AVX2Asm(SB), NOSPLIT, $0-20
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX

	TESTQ CX, CX
	JZ    max_bf16_avx2_zero

	MOVWLZX (SI), AX
	SHLQ  $16, AX
	VMOVD X0, AX
	VBROADCASTSS X0, Y0

	ADDQ $2, SI
	DECQ CX

max_bf16_avx2_w8:
	CMPQ CX, $8
	JL    max_bf16_avx2_w4

	WIDEN_BF16_8H_AVX2(SI, Y1)
	VMAXPS Y1, Y0, Y0

	ADDQ $16, SI
	SUBQ $8, CX
	JMP  max_bf16_avx2_w8

max_bf16_avx2_w4:
	CMPQ CX, $4
	JL    max_bf16_avx2_tail

	VMOVDQU X1, (SI)
	VPXOR X3, X3, X3
	VPUNPCKLWD X3, X1, X4
	VPUNPCKHWD X3, X1, X5
	VPSLLD $16, X4, X4
	VPSLLD $16, X5, X5
	VINSERTF128 $1, X5, Y4, Y1
	VMAXPS Y1, Y0, Y0

	ADDQ $8, SI
	SUBQ $4, CX
	JMP  max_bf16_avx2_w4

max_bf16_avx2_tail:
	BF16_MAX_REDUCE_AVX2

	TESTQ CX, CX
	JZ    max_bf16_avx2_store

max_bf16_avx2_scalar:
	MOVWLZX (SI), AX
	SHLQ  $16, AX
	VMOVD X1, AX
	VMAXSS X1, X0, X0

	ADDQ $2, SI
	DECQ CX
	JNZ  max_bf16_avx2_scalar

max_bf16_avx2_store:
	STORE_XMM0_F32

max_bf16_avx2_zero:
	XORPS X0, X0
	MOVSS X0, ret+16(FP)
	RET

// func L1NormBFloat16AVX2Asm(src *uint16, count int) float32
TEXT ·L1NormBFloat16AVX2Asm(SB), NOSPLIT, $0-20
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX

	TESTQ CX, CX
	JZ    l1_bf16_avx2_zero

	VXORPS Y0, Y0, Y0
	VMOVUPS l1ReducedAbsMaskAVX2<>(SB), X6
	VINSERTF128 $1, X6, Y6, Y6

l1_bf16_avx2_w8:
	CMPQ CX, $8
	JL    l1_bf16_avx2_w4

	WIDEN_BF16_8H_AVX2(SI, Y1)
	VANDPS Y6, Y1, Y1
	VADDPS Y1, Y0, Y0

	ADDQ $16, SI
	SUBQ $8, CX
	JMP  l1_bf16_avx2_w8

l1_bf16_avx2_w4:
	CMPQ CX, $4
	JL    l1_bf16_avx2_tail

	VMOVDQU X1, (SI)
	VPXOR X3, X3, X3
	VPUNPCKLWD X3, X1, X4
	VPUNPCKHWD X3, X1, X5
	VPSLLD $16, X4, X4
	VPSLLD $16, X5, X5
	VINSERTF128 $1, X5, Y4, Y1
	VANDPS Y6, Y1, Y1
	VADDPS Y1, Y0, Y0

	ADDQ $8, SI
	SUBQ $4, CX
	JMP  l1_bf16_avx2_w4

l1_bf16_avx2_tail:
	BF16_L1_REDUCE_AVX2

	TESTQ CX, CX
	JZ    l1_bf16_avx2_store

l1_bf16_avx2_scalar:
	MOVWLZX (SI), AX
	SHLQ  $16, AX
	VMOVD X1, AX
	VANDPS X6, X1, X1
	VADDSS X1, X0, X0

	ADDQ $2, SI
	DECQ CX
	JNZ  l1_bf16_avx2_scalar

l1_bf16_avx2_store:
	STORE_XMM0_F32

l1_bf16_avx2_zero:
	XORPS X0, X0
	MOVSS X0, ret+16(FP)
	RET

// func MinFloat16AVX2Asm(src *uint16, count int) float32
TEXT ·MinFloat16AVX2Asm(SB), NOSPLIT, $0-20
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX

	TESTQ CX, CX
	JZ    min_fp16_avx2_zero

	MOVWLZX (SI), AX
	VMOVD X0, AX
	VCVTPH2PS X0, X0
	VBROADCASTSS X0, Y0

	ADDQ $2, SI
	DECQ CX

min_fp16_avx2_w8:
	CMPQ CX, $8
	JL    min_fp16_avx2_w4

	WIDEN_FP16_8H_AVX2(SI, Y1)
	VMINPS Y1, Y0, Y0

	ADDQ $16, SI
	SUBQ $8, CX
	JMP  min_fp16_avx2_w8

min_fp16_avx2_w4:
	CMPQ CX, $4
	JL    min_fp16_avx2_tail

	VMOVDQU X1, (SI)
	VCVTPH2PS X1, Y1
	VMINPS Y1, Y0, Y0

	ADDQ $8, SI
	SUBQ $4, CX
	JMP  min_fp16_avx2_w4

min_fp16_avx2_tail:
	BF16_MINMAX_REDUCE_AVX2

	TESTQ CX, CX
	JZ    min_fp16_avx2_store

min_fp16_avx2_scalar:
	MOVWLZX (SI), AX
	VMOVD X1, AX
	VCVTPH2PS X1, X1
	VMINSS X1, X0, X0

	ADDQ $2, SI
	DECQ CX
	JNZ  min_fp16_avx2_scalar

min_fp16_avx2_store:
	STORE_XMM0_F32

min_fp16_avx2_zero:
	XORPS X0, X0
	MOVSS X0, ret+16(FP)
	RET

// func MaxFloat16AVX2Asm(src *uint16, count int) float32
TEXT ·MaxFloat16AVX2Asm(SB), NOSPLIT, $0-20
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX

	TESTQ CX, CX
	JZ    max_fp16_avx2_zero

	MOVWLZX (SI), AX
	VMOVD X0, AX
	VCVTPH2PS X0, X0
	VBROADCASTSS X0, Y0

	ADDQ $2, SI
	DECQ CX

max_fp16_avx2_w8:
	CMPQ CX, $8
	JL    max_fp16_avx2_w4

	WIDEN_FP16_8H_AVX2(SI, Y1)
	VMAXPS Y1, Y0, Y0

	ADDQ $16, SI
	SUBQ $8, CX
	JMP  max_fp16_avx2_w8

max_fp16_avx2_w4:
	CMPQ CX, $4
	JL    max_fp16_avx2_tail

	VMOVDQU X1, (SI)
	VCVTPH2PS X1, Y1
	VMAXPS Y1, Y0, Y0

	ADDQ $8, SI
	SUBQ $4, CX
	JMP  max_fp16_avx2_w4

max_fp16_avx2_tail:
	BF16_MAX_REDUCE_AVX2

	TESTQ CX, CX
	JZ    max_fp16_avx2_store

max_fp16_avx2_scalar:
	MOVWLZX (SI), AX
	VMOVD X1, AX
	VCVTPH2PS X1, X1
	VMAXSS X1, X0, X0

	ADDQ $2, SI
	DECQ CX
	JNZ  max_fp16_avx2_scalar

max_fp16_avx2_store:
	STORE_XMM0_F32

max_fp16_avx2_zero:
	XORPS X0, X0
	MOVSS X0, ret+16(FP)
	RET

// func L1NormFloat16AVX2Asm(src *uint16, count int) float32
TEXT ·L1NormFloat16AVX2Asm(SB), NOSPLIT, $0-20
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX

	TESTQ CX, CX
	JZ    l1_fp16_avx2_zero

	VXORPS Y0, Y0, Y0
	VMOVUPS l1ReducedAbsMaskAVX2<>(SB), X6
	VINSERTF128 $1, X6, Y6, Y6

l1_fp16_avx2_w8:
	CMPQ CX, $8
	JL    l1_fp16_avx2_w4

	WIDEN_FP16_8H_AVX2(SI, Y1)
	VANDPS Y6, Y1, Y1
	VADDPS Y1, Y0, Y0

	ADDQ $16, SI
	SUBQ $8, CX
	JMP  l1_fp16_avx2_w8

l1_fp16_avx2_w4:
	CMPQ CX, $4
	JL    l1_fp16_avx2_tail

	VMOVDQU X1, (SI)
	VCVTPH2PS X1, Y1
	VANDPS Y6, Y1, Y1
	VADDPS Y1, Y0, Y0

	ADDQ $8, SI
	SUBQ $4, CX
	JMP  l1_fp16_avx2_w4

l1_fp16_avx2_tail:
	BF16_L1_REDUCE_AVX2

	TESTQ CX, CX
	JZ    l1_fp16_avx2_store

l1_fp16_avx2_scalar:
	MOVWLZX (SI), AX
	VMOVD X1, AX
	VCVTPH2PS X1, X1
	VANDPS X6, X1, X1
	VADDSS X1, X0, X0

	ADDQ $2, SI
	DECQ CX
	JNZ  l1_fp16_avx2_scalar

l1_fp16_avx2_store:
	STORE_XMM0_F32

l1_fp16_avx2_zero:
	XORPS X0, X0
	MOVSS X0, ret+16(FP)
	RET
