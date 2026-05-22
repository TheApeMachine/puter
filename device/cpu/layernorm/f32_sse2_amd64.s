#include "textflag.h"

// func LayerNormSquaredDiffSumFloat32SSE2Asm(row *float32, count int, mean float32) float32
TEXT ·LayerNormSquaredDiffSumFloat32SSE2Asm(SB), NOSPLIT, $0-28
	MOVQ row+0(FP), SI
	MOVQ count+8(FP), CX
	MOVSS mean+16(FP), X8
	VBROADCASTSS X8, X8

	TESTQ CX, CX
	JZ   ln_ssd_sse2_zero

	XORPD X0, X0

ln_ssd_sse2_w4:
	CMPQ CX, $4
	JL   ln_ssd_sse2_tail

	VMOVUPS (SI), X1
	VSUBPS  X8, X1, X2
	VCVTPS2PD X2, X3
	VMULPD  X3, X3, X3
	ADDPD   X3, X0

	ADDQ $16, SI
	SUBQ $4, CX
	JMP  ln_ssd_sse2_w4

ln_ssd_sse2_tail:
	TESTQ CX, CX
	JZ   ln_ssd_sse2_reduce

ln_ssd_sse2_scalar:
	MOVSS (SI), X1
	VSUBSS X8, X1, X2
	CVTSS2SD X2, X2
	MULSD  X2, X2
	ADDSD  X2, X0
	ADDQ  $4, SI
	DECQ  CX
	JNZ  ln_ssd_sse2_scalar

ln_ssd_sse2_reduce:
	ADDSD X0, X0
	MOVSD X0, X1
	UNPCKHPD X0, X1
	ADDSD X1, X0
	CVTSD2SS X0, X0
	MOVSS X0, ret+24(FP)
	RET

ln_ssd_sse2_zero:
	XORPS X0, X0
	MOVSS X0, ret+24(FP)
	RET

// func LayerNormApplyRowFloat32SSE2Asm(out, row, scale, bias *float32, count int, mean, invStdDev float32)
TEXT ·LayerNormApplyRowFloat32SSE2Asm(SB), NOSPLIT, $0-48
	MOVQ out+0(FP), DI
	MOVQ row+8(FP), SI
	MOVQ scale+16(FP), R8
	MOVQ bias+24(FP), R9
	MOVQ count+32(FP), CX
	MOVSS mean+40(FP), X10
	VBROADCASTSS X10, X10
	MOVSS invStdDev+44(FP), X11
	VBROADCASTSS X11, X11

ln_apply_sse2_w4:
	CMPQ CX, $4
	JL   ln_apply_sse2_tail

	VMOVUPS (SI), X1
	VMOVUPS (R8), X2
	VMOVUPS (R9), X3
	VSUBPS  X10, X1, X4
	VMULPS  X11, X4, X4
	VMULPS  X2, X4, X4
	VADDPS  X3, X4, X4
	VMOVUPS X4, (DI)

	ADDQ $16, SI
	ADDQ $16, DI
	ADDQ $16, R8
	ADDQ $16, R9
	SUBQ $4, CX
	JMP  ln_apply_sse2_w4

ln_apply_sse2_tail:
	TESTQ CX, CX
	JZ   ln_apply_sse2_done

ln_apply_sse2_scalar:
	MOVSS (SI), X1
	MOVSS (R8), X2
	MOVSS (R9), X3
	VSUBSS X10, X1, X4
	VMULSS X11, X4, X4
	VMULSS X2, X4, X4
	VADDSS X3, X4, X4
	MOVSS X4, (DI)
	ADDQ  $4, SI
	ADDQ  $4, DI
	ADDQ  $4, R8
	ADDQ  $4, R9
	DECQ  CX
	JNZ  ln_apply_sse2_scalar

ln_apply_sse2_done:
	RET
