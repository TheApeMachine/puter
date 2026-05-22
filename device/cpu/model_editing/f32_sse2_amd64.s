#include "textflag.h"

// func WeightGraftAddFloat32SSE2Asm(weights, injection *float32, count int)
TEXT ·WeightGraftAddFloat32SSE2Asm(SB), NOSPLIT, $0-24
	MOVQ weights+0(FP), DI
	MOVQ injection+8(FP), SI
	MOVQ count+16(FP), CX

mdl_sse2_w4:
	CMPQ CX, $4
	JL   mdl_sse2_tail

	VMOVUPS (DI), X0
	VMOVUPS (SI), X1
	VADDPS X1, X0, X0
	VMOVUPS X0, (DI)

	ADDQ $16, DI
	ADDQ $16, SI
	SUBQ $4, CX
	JMP  mdl_sse2_w4

mdl_sse2_tail:
	TESTQ CX, CX
	JZ   mdl_sse2_done

mdl_sse2_scalar:
	VMOVSS (DI), X0
	VMOVSS (SI), X1
	VADDSS X1, X0, X0
	MOVSS X0, (DI)
	ADDQ $4, DI
	ADDQ $4, SI
	DECQ CX
	JNZ  mdl_sse2_scalar

mdl_sse2_done:
	RET
