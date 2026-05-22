#include "textflag.h"

// func NormSquaredDiffSumFloat32AVX2Asm(row *float32, count int, mean float32) float32
TEXT ·NormSquaredDiffSumFloat32AVX2Asm(SB), NOSPLIT, $0-28
	MOVQ row+0(FP), SI
	MOVQ count+8(FP), CX
	MOVSS mean+16(FP), X8
	VBROADCASTSS X8, Y8

	TESTQ CX, CX
	JZ   norm_ssd_avx2_zero

	VXORPD Y0, Y0, Y0

norm_ssd_avx2_w8:
	CMPQ CX, $8
	JL   norm_ssd_avx2_w4

	VMOVUPS Y1, (SI)
	VSUBPS  Y8, Y1, Y2
	VEXTRACTF128 $0, Y2, X3
	VCVTPS2PD X3, Y4
	VMULPD  Y4, Y4, Y4
	VADDPD  Y0, Y4, Y0
	VEXTRACTF128 $1, Y2, X3
	VCVTPS2PD X3, Y4
	VMULPD  Y4, Y4, Y4
	VADDPD  Y0, Y4, Y0

	ADDQ $32, SI
	SUBQ $8, CX
	JMP  norm_ssd_avx2_w8

norm_ssd_avx2_w4:
	CMPQ CX, $4
	JL   norm_ssd_avx2_tail

	VMOVUPS X1, (SI)
	VSUBPS  X8, X1, X2
	VCVTPS2PD X2, Y4
	VMULPD  Y4, Y4, Y4
	VADDPD  Y0, Y4, Y0

	ADDQ $16, SI
	SUBQ $4, CX
	JMP  norm_ssd_avx2_w4

norm_ssd_avx2_tail:
	TESTQ CX, CX
	JZ   norm_ssd_avx2_reduce

norm_ssd_avx2_scalar:
	MOVSS (SI), X1
	VSUBSS X8, X1, X2
	CVTSS2SD X2, X2
	MULSD  X2, X2
	ADDSD  X2, X0
	ADDQ  $4, SI
	DECQ  CX
	JNZ  norm_ssd_avx2_scalar

norm_ssd_avx2_reduce:
	VHADDPD Y1, Y0, Y0
	VHADDPD Y1, Y1, Y1
	VEXTRACTF128 $0, Y1, X0
	CVTSD2SS X0, X0
	MOVSS X0, ret+24(FP)
	RET

norm_ssd_avx2_zero:
	XORPS X0, X0
	MOVSS X0, ret+24(FP)
	RET

// func NormApplyConstScaleBiasFloat32AVX2Asm(out, row *float32, count int, mean, invStdDev, scale, bias float32)
TEXT ·NormApplyConstScaleBiasFloat32AVX2Asm(SB), NOSPLIT, $0-44
	MOVQ out+0(FP), DI
	MOVQ row+8(FP), SI
	MOVQ count+16(FP), CX
	MOVSS mean+24(FP), X10
	VBROADCASTSS X10, Y10
	MOVSS invStdDev+28(FP), X11
	VBROADCASTSS X11, Y11
	MOVSS scale+32(FP), X12
	VBROADCASTSS X12, Y12
	MOVSS bias+36(FP), X13
	VBROADCASTSS X13, Y13

norm_apply_avx2_w8:
	CMPQ CX, $8
	JL   norm_apply_avx2_w4

	VMOVUPS (SI), Y1
	VSUBPS  Y10, Y1, Y4
	VMULPS  Y11, Y4, Y4
	VMULPS  Y12, Y4, Y4
	VADDPS  Y13, Y4, Y4
	VMOVUPS Y4, (DI)

	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $8, CX
	JMP  norm_apply_avx2_w8

norm_apply_avx2_w4:
	CMPQ CX, $4
	JL   norm_apply_avx2_tail

	VMOVUPS (SI), X1
	VSUBPS  X10, X1, X4
	VMULPS  X11, X4, X4
	VMULPS  X12, X4, X4
	VADDPS  X13, X4, X4
	VMOVUPS X4, (DI)

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  norm_apply_avx2_w4

norm_apply_avx2_tail:
	TESTQ CX, CX
	JZ   norm_apply_avx2_done

norm_apply_avx2_scalar:
	MOVSS (SI), X1
	VSUBSS X10, X1, X4
	VMULSS X11, X4, X4
	VMULSS X12, X4, X4
	VADDSS X13, X4, X4
	MOVSS X4, (DI)
	ADDQ  $4, SI
	ADDQ  $4, DI
	DECQ  CX
	JNZ  norm_apply_avx2_scalar

norm_apply_avx2_done:
	RET
