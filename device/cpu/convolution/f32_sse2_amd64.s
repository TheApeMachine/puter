#include "textflag.h"

// func ConvPatchDotFloat32SSE2Asm(weight, patch *float32, length int) float32
TEXT ·ConvPatchDotFloat32SSE2Asm(SB), NOSPLIT, $0-28
	MOVQ weight+0(FP), SI
	MOVQ patch+8(FP), DI
	MOVQ length+16(FP), CX

	TESTQ CX, CX
	JZ   cpd_sse2_zero

	XORPD X0, X0

cpd_sse2_w4:
	CMPQ CX, $4
	JL   cpd_sse2_tail

	VMOVUPS (SI), X1
	VMOVUPS (DI), X2
	VCVTPS2PD X1, X3
	VCVTPS2PD X2, X4
	MULPD   X4, X3
	ADDPD   X3, X0

	MOVAPS X1, X5
	SHUFPS $0xEE, X1, X5
	MOVAPS X2, X6
	SHUFPS $0xEE, X2, X6
	VCVTPS2PD X5, X3
	VCVTPS2PD X6, X4
	MULPD   X4, X3
	ADDPD   X3, X0

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  cpd_sse2_w4

cpd_sse2_tail:
	TESTQ CX, CX
	JZ   cpd_sse2_reduce

cpd_sse2_scalar:
	MOVSS (SI), X1
	MOVSS (DI), X2
	CVTSS2SD X1, X1
	CVTSS2SD X2, X2
	MULSD X2, X1
	ADDSD X1, X0
	ADDQ  $4, SI
	ADDQ  $4, DI
	DECQ  CX
	JNZ  cpd_sse2_scalar

cpd_sse2_reduce:
	MOVAPD X0, X1
	SHUFPD $1, X0, X1
	ADDPD  X1, X0
	CVTSD2SS X0, X0
	MOVSS X0, ret+24(FP)
	RET

cpd_sse2_zero:
	XORPS X0, X0
	MOVSS X0, ret+24(FP)
	RET
