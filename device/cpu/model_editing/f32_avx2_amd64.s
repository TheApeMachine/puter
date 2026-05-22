#include "textflag.h"

// func WeightGraftAddFloat32AVX2Asm(weights, injection *float32, count int)
TEXT ·WeightGraftAddFloat32AVX2Asm(SB), NOSPLIT, $0-24
	MOVQ weights+0(FP), DI
	MOVQ injection+8(FP), SI
	MOVQ count+16(FP), CX

mdl_avx2_w8:
	CMPQ CX, $8
	JL   mdl_avx2_w4

	VMOVUPS (DI), Y0
	VMOVUPS (SI), Y1
	VADDPS Y1, Y0, Y0
	VMOVUPS Y0, (DI)

	ADDQ $32, DI
	ADDQ $32, SI
	SUBQ $8, CX
	JMP  mdl_avx2_w8

mdl_avx2_w4:
	CMPQ CX, $4
	JL   mdl_avx2_tail

	VMOVUPS (DI), X0
	VMOVUPS (SI), X1
	VADDPS X1, X0, X0
	VMOVUPS X0, (DI)

	ADDQ $16, DI
	ADDQ $16, SI
	SUBQ $4, CX
	JMP  mdl_avx2_w4

mdl_avx2_tail:
	TESTQ CX, CX
	JZ   mdl_avx2_done

mdl_avx2_scalar:
	VMOVSS (DI), X0
	VMOVSS (SI), X1
	VADDSS X1, X0, X0
	MOVSS X0, (DI)
	ADDQ $4, DI
	ADDQ $4, SI
	DECQ CX
	JNZ  mdl_avx2_scalar

mdl_avx2_done:
	RET
