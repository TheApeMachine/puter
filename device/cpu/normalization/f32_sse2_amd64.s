#include "textflag.h"

// func NormSquaredDiffSumFloat32SSE2Asm(row *float32, count int, mean float32) float32
TEXT ·NormSquaredDiffSumFloat32SSE2Asm(SB), NOSPLIT, $0-28
	MOVQ row+0(FP), SI
	MOVQ count+8(FP), CX
	MOVSS mean+16(FP), X8
	VBROADCASTSS X8, X8

	TESTQ CX, CX
	JZ   norm_ssd_sse2_zero

	XORPD X0, X0

norm_ssd_sse2_w4:
	CMPQ CX, $4
	JL   norm_ssd_sse2_tail

	VMOVUPS (SI), X1
	VSUBPS  X8, X1, X2
	VCVTPS2PD X2, X3
	VMULPD  X3, X3, X3
	ADDPD   X3, X0

	ADDQ $16, SI
	SUBQ $4, CX
	JMP  norm_ssd_sse2_w4

norm_ssd_sse2_tail:
	TESTQ CX, CX
	JZ   norm_ssd_sse2_reduce

norm_ssd_sse2_scalar:
	MOVSS (SI), X1
	VSUBSS X8, X1, X2
	CVTSS2SD X2, X2
	MULSD  X2, X2
	ADDSD  X2, X0
	ADDQ  $4, SI
	DECQ  CX
	JNZ  norm_ssd_sse2_scalar

norm_ssd_sse2_reduce:
	ADDSD X0, X0
	MOVSD X0, X1
	UNPCKHPD X0, X1
	ADDSD X1, X0
	CVTSD2SS X0, X0
	MOVSS X0, ret+24(FP)
	RET

norm_ssd_sse2_zero:
	XORPS X0, X0
	MOVSS X0, ret+24(FP)
	RET

// func NormApplyConstScaleBiasFloat32SSE2Asm(out, row *float32, count int, mean, invStdDev, scale, bias float32)
TEXT ·NormApplyConstScaleBiasFloat32SSE2Asm(SB), NOSPLIT, $0-44
	MOVQ out+0(FP), DI
	MOVQ row+8(FP), SI
	MOVQ count+16(FP), CX
	MOVSS mean+24(FP), X10
	VBROADCASTSS X10, X10
	MOVSS invStdDev+28(FP), X11
	VBROADCASTSS X11, X11
	MOVSS scale+32(FP), X12
	VBROADCASTSS X12, X12
	MOVSS bias+36(FP), X13
	VBROADCASTSS X13, X13

norm_apply_sse2_w4:
	CMPQ CX, $4
	JL   norm_apply_sse2_tail

	VMOVUPS (SI), X1
	VSUBPS  X10, X1, X4
	VMULPS  X11, X4, X4
	VMULPS  X12, X4, X4
	VADDPS  X13, X4, X4
	VMOVUPS X4, (DI)

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  norm_apply_sse2_w4

norm_apply_sse2_tail:
	TESTQ CX, CX
	JZ   norm_apply_sse2_done

norm_apply_sse2_scalar:
	MOVSS (SI), X1
	VSUBSS X10, X1, X4
	VMULSS X11, X4, X4
	VMULSS X12, X4, X4
	VADDSS X13, X4, X4
	MOVSS X4, (DI)
	ADDQ  $4, SI
	ADDQ  $4, DI
	DECQ  CX
	JNZ  norm_apply_sse2_scalar

norm_apply_sse2_done:
	RET
