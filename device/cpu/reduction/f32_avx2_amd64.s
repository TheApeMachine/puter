#include "textflag.h"

DATA redAbsMaskAVX2<>+0(SB)/4, $0x7fffffff
DATA redAbsMaskAVX2<>+4(SB)/4, $0x7fffffff
DATA redAbsMaskAVX2<>+8(SB)/4, $0x7fffffff
DATA redAbsMaskAVX2<>+12(SB)/4, $0x7fffffff
GLOBL redAbsMaskAVX2<>(SB), RODATA|NOPTR, $16

DATA redOneF32AVX2<>+0(SB)/4, $0x3f800000
GLOBL redOneF32AVX2<>(SB), RODATA|NOPTR, $4

// func SumFloat32AVX2Asm(src *float32, count int) float32
TEXT ·SumFloat32AVX2Asm(SB), NOSPLIT, $0-20
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX
	TESTQ CX, CX
	JZ   sum_avx2_zero

	VXORPD Y0, Y0, Y0

sum_avx2_w8:
	CMPQ CX, $8
	JL   sum_avx2_w4

	VMOVUPS (SI), Y2
	VEXTRACTF128 $0, Y2, X3
	VCVTPS2PD X3, Y4
	VADDPD  Y0, Y4, Y0
	VEXTRACTF128 $1, Y2, X3
	VCVTPS2PD X3, Y4
	VADDPD  Y0, Y4, Y0

	ADDQ $32, SI
	SUBQ $8, CX
	JMP  sum_avx2_w8

sum_avx2_w4:
	CMPQ CX, $4
	JL   sum_avx2_tail

	VMOVUPS (SI), X2
	VCVTPS2PD X2, Y3
	VADDPD  Y0, Y3, Y0

	ADDQ $16, SI
	SUBQ $4, CX
	JMP  sum_avx2_w4

sum_avx2_tail:
	TESTQ CX, CX
	JZ   sum_avx2_reduce

sum_avx2_scalar:
	MOVSS (SI), X2
	CVTSS2SD X2, X2
	ADDSD  X2, X0
	ADDQ  $4, SI
	DECQ  CX
	JNZ  sum_avx2_scalar

sum_avx2_reduce:
	VHADDPD Y0, Y0, Y0
	VEXTRACTF128 $0, Y0, X0
	VHADDPD X0, X0, X0
	CVTSD2SS X0, X0
	MOVSS X0, ret+16(FP)
	RET

sum_avx2_zero:
	XORPS X0, X0
	MOVSS X0, ret+16(FP)
	RET

// func ProdFloat32AVX2Asm(src *float32, count int) float32
TEXT ·ProdFloat32AVX2Asm(SB), NOSPLIT, $0-20
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX
	TESTQ CX, CX
	JZ   prod_avx2_zero

	VBROADCASTSS redOneF32AVX2<>(SB), Y0

prod_avx2_w8:
	CMPQ CX, $8
	JL   prod_avx2_w4

	VMOVUPS (SI), Y1
	VMULPS  Y1, Y0, Y0

	ADDQ $32, SI
	SUBQ $8, CX
	JMP  prod_avx2_w8

prod_avx2_w4:
	CMPQ CX, $4
	JL   prod_avx2_tail

	VMOVUPS (SI), X1
	VMULPS  X1, X0, X0

	ADDQ $16, SI
	SUBQ $4, CX
	JMP  prod_avx2_w4

prod_avx2_tail:
	TESTQ CX, CX
	JZ   prod_avx2_fold

prod_avx2_scalar:
	MOVSS (SI), X1
	MULSS X1, X0
	ADDQ  $4, SI
	DECQ  CX
	JNZ  prod_avx2_scalar

prod_avx2_fold:
	VEXTRACTF128 $0, Y0, X0
	VEXTRACTF128 $1, Y0, X1
	VMULPS  X1, X0, X0
	MOVAPS  X0, X1
	SHUFPS  $2, X0, X1
	MULPS   X1, X0
	MOVAPS  X0, X1
	SHUFPS  $1, X0, X1
	MULPS   X1, X0
	MOVSS X0, ret+16(FP)
	RET

prod_avx2_zero:
	MOVSS redOneF32AVX2<>(SB), X0
	MOVSS X0, ret+16(FP)
	RET

// func ReduceMaxFloat32AVX2Asm(src *float32, count int) float32
TEXT ·ReduceMaxFloat32AVX2Asm(SB), NOSPLIT, $0-20
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX
	TESTQ CX, CX
	JZ   max_avx2_zero

	MOVSS (SI), X0
	VBROADCASTSS X0, Y0
	ADDQ $4, SI
	DECQ CX

max_avx2_w8:
	CMPQ CX, $8
	JL   max_avx2_w4

	VMOVUPS (SI), Y1
	VMAXPS  Y1, Y0, Y0

	ADDQ $32, SI
	SUBQ $8, CX
	JMP  max_avx2_w8

max_avx2_w4:
	CMPQ CX, $4
	JL   max_avx2_tail

	VMOVUPS (SI), X1
	VMAXPS  X1, X0, X0

	ADDQ $16, SI
	SUBQ $4, CX
	JMP  max_avx2_w4

max_avx2_tail:
	TESTQ CX, CX
	JZ   max_avx2_fold

max_avx2_scalar:
	MOVSS (SI), X1
	MAXSS X1, X0
	ADDQ  $4, SI
	DECQ  CX
	JNZ  max_avx2_scalar

max_avx2_fold:
	VEXTRACTF128 $0, Y0, X0
	VEXTRACTF128 $1, Y0, X1
	VMAXPS  X1, X0, X0
	MOVAPS  X0, X1
	SHUFPS  $2, X0, X1
	MAXPS   X1, X0
	MOVAPS  X0, X1
	SHUFPS  $1, X0, X1
	MAXPS   X1, X0
	MOVSS X0, ret+16(FP)
	RET

max_avx2_zero:
	XORPS X0, X0
	MOVSS X0, ret+16(FP)
	RET

// func ReduceMinFloat32AVX2Asm(src *float32, count int) float32
TEXT ·ReduceMinFloat32AVX2Asm(SB), NOSPLIT, $0-20
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX
	TESTQ CX, CX
	JZ   min_avx2_zero

	MOVSS (SI), X0
	VBROADCASTSS X0, Y0
	ADDQ $4, SI
	DECQ CX

min_avx2_w8:
	CMPQ CX, $8
	JL   min_avx2_w4

	VMOVUPS (SI), Y1
	VMINPS  Y1, Y0, Y0

	ADDQ $32, SI
	SUBQ $8, CX
	JMP  min_avx2_w8

min_avx2_w4:
	CMPQ CX, $4
	JL   min_avx2_tail

	VMOVUPS (SI), X1
	VMINPS  X1, X0, X0

	ADDQ $16, SI
	SUBQ $4, CX
	JMP  min_avx2_w4

min_avx2_tail:
	TESTQ CX, CX
	JZ   min_avx2_fold

min_avx2_scalar:
	MOVSS (SI), X1
	MINSS X1, X0
	ADDQ  $4, SI
	DECQ  CX
	JNZ  min_avx2_scalar

min_avx2_fold:
	VEXTRACTF128 $0, Y0, X0
	VEXTRACTF128 $1, Y0, X1
	VMINPS  X1, X0, X0
	MOVAPS  X0, X1
	SHUFPS  $2, X0, X1
	MINPS   X1, X0
	MOVAPS  X0, X1
	SHUFPS  $1, X0, X1
	MINPS   X1, X0
	MOVSS X0, ret+16(FP)
	RET

min_avx2_zero:
	XORPS X0, X0
	MOVSS X0, ret+16(FP)
	RET

// func L1NormFloat32AVX2Asm(src *float32, count int) float32
TEXT ·L1NormFloat32AVX2Asm(SB), NOSPLIT, $0-20
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX
	TESTQ CX, CX
	JZ   l1_avx2_zero

	VXORPD Y0, Y0, Y0
	VMOVUPS redAbsMaskAVX2<>(SB), X6

l1_avx2_w8:
	CMPQ CX, $8
	JL   l1_avx2_w4

	VMOVUPS (SI), Y2
	VEXTRACTF128 $0, Y2, X3
	VANDPS  X6, X3, X3
	VCVTPS2PD X3, Y4
	VADDPD  Y0, Y4, Y0
	VEXTRACTF128 $1, Y2, X3
	VANDPS  X6, X3, X3
	VCVTPS2PD X3, Y4
	VADDPD  Y0, Y4, Y0

	ADDQ $32, SI
	SUBQ $8, CX
	JMP  l1_avx2_w8

l1_avx2_w4:
	CMPQ CX, $4
	JL   l1_avx2_tail

	VMOVUPS (SI), X2
	VANDPS  X6, X2, X2
	VCVTPS2PD X2, Y3
	VADDPD  Y0, Y3, Y0

	ADDQ $16, SI
	SUBQ $4, CX
	JMP  l1_avx2_w4

l1_avx2_tail:
	TESTQ CX, CX
	JZ   l1_avx2_reduce

l1_avx2_scalar:
	MOVSS (SI), X2
	ANDPS X6, X2
	CVTSS2SD X2, X2
	ADDSD  X2, X0
	ADDQ  $4, SI
	DECQ  CX
	JNZ  l1_avx2_scalar

l1_avx2_reduce:
	VHADDPD Y0, Y0, Y0
	VEXTRACTF128 $0, Y0, X0
	VHADDPD X0, X0, X0
	CVTSD2SS X0, X0
	MOVSS X0, ret+16(FP)
	RET

l1_avx2_zero:
	XORPS X0, X0
	MOVSS X0, ret+16(FP)
	RET
