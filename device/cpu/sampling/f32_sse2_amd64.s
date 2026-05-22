#include "textflag.h"

DATA samSse2OneF32<>+0(SB)/4, $0x3f800000
GLOBL samSse2OneF32<>(SB), RODATA|NOPTR, $4

DATA samSse2ExpC<>+0(SB)/4, $1.4426950408889634
DATA samSse2ExpC<>+4(SB)/4, $0.6931471805599453
DATA samSse2ExpC<>+12(SB)/4, $0.00019841270
DATA samSse2ExpC<>+16(SB)/4, $0.0013888889
DATA samSse2ExpC<>+20(SB)/4, $0.008333334
DATA samSse2ExpC<>+24(SB)/4, $0.041666667
DATA samSse2ExpC<>+28(SB)/4, $0.16666667
DATA samSse2ExpC<>+32(SB)/4, $0.5
DATA samSse2ExpC<>+36(SB)/4, $1.0
DATA samSse2ExpC<>+40(SB)/4, $1.0
GLOBL samSse2ExpC<>(SB), RODATA|NOPTR, $44

DATA samSse2ExpBias127<>+0(SB)/4, $127
GLOBL samSse2ExpBias127<>(SB), RODATA|NOPTR, $4

DATA samSse2SoftmaxClamp<>+0(SB)/4, $-87.0
GLOBL samSse2SoftmaxClamp<>(SB), RODATA|NOPTR, $4

// func GreedySampleFloat32SSE2Asm(logits *float32, count int) int32
TEXT ·GreedySampleFloat32SSE2Asm(SB), NOSPLIT, $0-20
	MOVQ logits+0(FP), SI
	MOVQ SI, BX
	MOVQ count+8(FP), CX

	TESTQ CX, CX
	JZ   greedy_sse2_zero

	CMPQ CX, $1
	JE   greedy_sse2_one

	MOVSS (SI), X0
	SHUFPS $0x00, X0, X0
	ADDQ $4, SI
	DECQ CX

greedy_sse2_max_w4:
	CMPQ CX, $4
	JL   greedy_sse2_max_tail

	MOVUPS (SI), X4
	MAXPS X4, X0

	ADDQ $16, SI
	SUBQ $4, CX
	JMP  greedy_sse2_max_w4

greedy_sse2_max_tail:
	TESTQ CX, CX
	JZ   greedy_sse2_max_done

greedy_sse2_max_scalar:
	MOVSS (SI), X4
	MAXSS X4, X0
	ADDQ  $4, SI
	DECQ  CX
	JNZ  greedy_sse2_max_scalar

greedy_sse2_max_done:
	MOVAPS X0, X4
	SHUFPS $0x4E, X0, X4
	MAXPS  X4, X0
	MOVAPS X0, X4
	SHUFPS $0xB1, X0, X4
	MAXPS  X4, X0
	MOVSS  X0, X0

	MOVQ BX, SI
	MOVQ count+8(FP), CX
	XORQ R8, R8

greedy_sse2_find_scalar:
	CMPQ R8, CX
	JGE  greedy_sse2_fail

	MOVSS (SI), X4
	UCOMISS X0, X4
	JNE  greedy_sse2_find_next
	MOVL R8, ret+16(FP)
	RET

greedy_sse2_find_next:
	ADDQ $4, SI
	INCQ R8
	JMP  greedy_sse2_find_scalar

greedy_sse2_fail:
	MOVQ count+8(FP), AX
	DECQ AX
	MOVL AX, ret+16(FP)
	RET

greedy_sse2_one:
	XORL AX, AX
	MOVL AX, ret+16(FP)
	RET

greedy_sse2_zero:
	XORL AX, AX
	MOVL AX, ret+16(FP)
	RET

// func SamplingSoftmaxRowFloat32SSE2Asm(logits, out *float32, temperature float32, count int)
TEXT ·SamplingSoftmaxRowFloat32SSE2Asm(SB), NOSPLIT, $32-28
	MOVQ logits+0(FP), SI
	MOVQ out+8(FP), DI
	MOVSS temperature+16(FP), X10
	MOVQ count+24(FP), CX
	TESTQ CX, CX
	JZ   sam_sse2_softmax_done

	XORPS X11, X11
	UCOMISS X10, X11
	JNE  sam_sse2_softmax_temp_ok
	MOVSS samSse2OneF32<>(SB), X10

sam_sse2_softmax_temp_ok:
	SHUFPS $0, X10, X10

	MOVSS (SI), X0
	SHUFPS $0, X0, X0
	ADDQ $4, SI
	DECQ CX

sam_sse2_softmax_max_w4:
	CMPQ CX, $4
	JL   sam_sse2_softmax_max_reduce

	MOVUPS (SI), X1
	MAXPS  X1, X0

	ADDQ $16, SI
	SUBQ $4, CX
	JMP  sam_sse2_softmax_max_w4

sam_sse2_softmax_max_reduce:
	MOVAPS X0, X1
	SHUFPS $0x4E, X0, X1
	MAXPS  X1, X0
	MOVAPS X0, X1
	SHUFPS $0xB1, X0, X1
	MAXPS  X1, X0
	MOVSS  X0, X6
	SHUFPS $0, X6, X6

	TESTQ CX, CX
	JZ   sam_sse2_softmax_exp_setup

sam_sse2_softmax_max_scalar:
	MOVSS (SI), X1
	MAXSS X1, X0
	ADDQ  $4, SI
	DECQ  CX
	JNZ  sam_sse2_softmax_max_scalar

	MOVSS X0, X6
	SHUFPS $0, X6, X6

sam_sse2_softmax_exp_setup:
	MOVQ logits+0(FP), SI
	MOVQ out+8(FP), DI
	MOVQ count+24(FP), CX

	MOVQ $samSse2ExpC<>(SB), AX
	MOVSS (AX), X8
	SHUFPS $0, X8, X8
	MOVSS 4(AX), X9
	SHUFPS $0, X9, X9
	MOVSS 12(AX), X11
	SHUFPS $0, X11, X11
	MOVSS 16(AX), X12
	SHUFPS $0, X12, X12
	MOVSS 20(AX), X13
	SHUFPS $0, X13, X13
	MOVSS 24(AX), X14
	SHUFPS $0, X14, X14
	MOVSS 28(AX), X15
	SHUFPS $0, X15, X15
	MOVSS 32(AX), X0
	SHUFPS $0, X0, X0
	MOVAPS X0, sam_sse2_c8+0(SP)
	MOVSS 36(AX), X0
	SHUFPS $0, X0, X0
	MOVAPS X0, sam_sse2_c9+16(SP)
	MOVQ AX, R15
	MOVSS samSse2SoftmaxClamp<>(SB), X8
	SHUFPS $0, X8, X8
	MOVAPS X8, X4
	XORPS X5, X5

sam_sse2_softmax_exp_w4:
	CMPQ CX, $4
	JL   sam_sse2_softmax_exp_tail

	MOVUPS (SI), X0
	SUBPS X6, X0
	DIVPS X10, X0
	MAXPS X4, X0
	MOVAPS X0, X1
	MULPS X8, X1
	ROUNDPS $8, X1, X1
	MOVAPS X1, X2
	MULPS X9, X2
	SUBPS X2, X0
	MOVAPS X11, X3
	MULPS X0, X11
	ADDPS X3, X11
	MOVAPS X12, X3
	MULPS X0, X12
	ADDPS X3, X12
	MOVAPS X13, X3
	MULPS X0, X13
	ADDPS X3, X13
	MOVAPS X14, X3
	MULPS X0, X14
	ADDPS X3, X14
	MOVAPS X15, X3
	MULPS X0, X15
	ADDPS X3, X15
	MOVAPS X15, X7
	MULPS X0, X7
	ADDPS sam_sse2_c8+0(SP), X7
	MULPS X0, X7
	ADDPS sam_sse2_c9+16(SP), X7
	CVTPS2PL X1, X1
	MOVD samSse2ExpBias127<>(SB), X3
	PSHUFD $0, X3, X3
	PADDL X3, X1
	PSLLL $23, X1
	PADDL X1, X7
	ADDPS X7, X5
	MOVUPS X7, (DI)

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  sam_sse2_softmax_exp_w4

sam_sse2_softmax_exp_tail:
	TESTQ CX, CX
	JZ   sam_sse2_softmax_exp_reduce

	MOVSS 12(R15), X11
	MOVSS 16(R15), X12
	MOVSS 20(R15), X13
	MOVSS 24(R15), X14
	MOVSS 28(R15), X15
	MOVSS 32(R15), X0
	SHUFPS $0, X0, X0
	MOVAPS X0, sam_sse2_c8+0(SP)
	MOVSS 36(R15), X0
	SHUFPS $0, X0, X0
	MOVAPS X0, sam_sse2_c9+16(SP)
	MOVSS (AX), X8
	SHUFPS $0, X8, X8
	MOVSS 4(AX), X9
	SHUFPS $0, X9, X9

sam_sse2_softmax_exp_scalar:
	MOVSS (SI), X0
	SUBSS X6, X0
	DIVSS X10, X0
	MAXSS X4, X0
	MOVSS X0, X1
	MULSS X8, X1
	ROUNDSS $8, X1, X1
	MOVSS X1, X2
	MULSS X9, X2
	SUBSS X2, X0
	MOVSS X11, X3
	MULSS X0, X11
	ADDSS X3, X11
	MOVSS X12, X3
	MULSS X0, X12
	ADDSS X3, X12
	MOVSS X13, X3
	MULSS X0, X13
	ADDSS X3, X13
	MOVSS X14, X3
	MULSS X0, X14
	ADDSS X3, X14
	MOVSS X15, X3
	MULSS X0, X15
	ADDSS X3, X15
	MOVSS X15, X7
	MULSS X0, X7
	ADDSS sam_sse2_c8+0(SP), X7
	MULSS X0, X7
	ADDSS sam_sse2_c9+16(SP), X7
	XORPS X2, X2
	MOVSS X1, X2
	CVTPS2PL X2, X2
	MOVD samSse2ExpBias127<>(SB), X3
	PSHUFD $0, X3, X3
	PADDL X3, X2
	PSLLL $23, X2
	PADDL X2, X7
	ADDSS X7, X5
	MOVSS X7, (DI)
	ADDQ  $4, SI
	ADDQ  $4, DI
	DECQ  CX
	JNZ  sam_sse2_softmax_exp_scalar

sam_sse2_softmax_exp_reduce:
	MOVAPS X5, X0
	SHUFPS $1, X5, X5
	ADDPS X5, X0
	MOVAPS X0, X5
	SHUFPS $2, X0, X0
	ADDPS X5, X0
	XORPS X1, X1
	UCOMISS X0, X1
	JE    sam_sse2_softmax_done

	MOVSS samSse2OneF32<>(SB), X8
	DIVSS X0, X8
	SHUFPS $0, X8, X8

	MOVQ out+8(FP), DI
	MOVQ count+24(FP), CX

sam_sse2_softmax_scale_w4:
	CMPQ CX, $4
	JL   sam_sse2_softmax_scale_tail

	MOVUPS (DI), X0
	MULPS X8, X0
	MOVUPS X0, (DI)

	ADDQ $16, DI
	SUBQ $4, CX
	JMP  sam_sse2_softmax_scale_w4

sam_sse2_softmax_scale_tail:
	TESTQ CX, CX
	JZ   sam_sse2_softmax_done

sam_sse2_softmax_scale_scalar:
	MOVSS (DI), X0
	MULSS X8, X0
	MOVSS X0, (DI)
	ADDQ  $4, DI
	DECQ  CX
	JNZ  sam_sse2_softmax_scale_scalar

sam_sse2_softmax_done:
	RET
