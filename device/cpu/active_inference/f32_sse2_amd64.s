#include "textflag.h"

// func PrecisionWeightFloat32SSE2Asm(errors, precision, output *float32, count int)
TEXT ·PrecisionWeightFloat32SSE2Asm(SB), NOSPLIT, $0-32
	MOVQ errors+0(FP), SI
	MOVQ precision+8(FP), DX
	MOVQ output+16(FP), DI
	MOVQ count+24(FP), CX

	TESTQ CX, CX
	JZ   ai_sse2_pw_done

ai_sse2_pw_w4:
	CMPQ CX, $4
	JL   ai_sse2_pw_tail

	MOVUPS (SI), X0
	MOVUPS (DX), X1
	MULPS  X1, X0
	MOVUPS X0, (DI)

	ADDQ $16, SI
	ADDQ $16, DX
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  ai_sse2_pw_w4

ai_sse2_pw_tail:
	TESTQ CX, CX
	JZ   ai_sse2_pw_done

ai_sse2_pw_scalar:
	MOVSS (SI), X0
	MOVSS (DX), X1
	MULSS X1, X0
	MOVSS X0, (DI)
	ADDQ  $4, SI
	ADDQ  $4, DX
	ADDQ  $4, DI
	DECQ  CX
	JNZ   ai_sse2_pw_scalar

ai_sse2_pw_done:
	RET

// func BeliefUpdateFloat32SSE2Asm(likelihood, prior, output *float32, count int)
TEXT ·BeliefUpdateFloat32SSE2Asm(SB), NOSPLIT, $0-32
	MOVQ likelihood+0(FP), SI
	MOVQ prior+8(FP), DX
	MOVQ output+16(FP), DI
	MOVQ count+24(FP), CX

	TESTQ CX, CX
	JZ   ai_sse2_bu_done

	XORPD X0, X0

ai_sse2_bu_mul_w4:
	CMPQ CX, $4
	JL   ai_sse2_bu_mul_tail

	MOVUPS (SI), X1
	MOVUPS (DX), X2
	MULPS  X2, X1
	MOVUPS X1, (DI)
	CVTPS2PD X1, X3
	ADDPD  X3, X0
	MOVAPS X1, X4
	SHUFPS $0xEE, X1, X4
	CVTPS2PD X4, X3
	ADDPD  X3, X0

	ADDQ $16, SI
	ADDQ $16, DX
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  ai_sse2_bu_mul_w4

ai_sse2_bu_mul_tail:
	TESTQ CX, CX
	JZ   ai_sse2_bu_reduce

ai_sse2_bu_mul_scalar:
	MOVSS (SI), X1
	MOVSS (DX), X2
	MULSS X2, X1
	MOVSS X1, (DI)
	CVTSS2SD X1, X1
	ADDSD X1, X0
	ADDQ  $4, SI
	ADDQ  $4, DX
	ADDQ  $4, DI
	DECQ  CX
	JNZ   ai_sse2_bu_mul_scalar

ai_sse2_bu_reduce:
	MOVAPS X0, X1
	SHUFPD $1, X1, X1
	ADDPD  X1, X0
	CVTSD2SS X0, X0

	XORPS X1, X1
	UCOMISS X0, X1
	JZ    ai_sse2_bu_done

	MOVSS aiOneBits<>(SB), X3
	DIVSS X2, X3
	SHUFPS $0, X3, X3

	MOVQ output+16(FP), DI
	MOVQ count+24(FP), CX

ai_sse2_bu_scale_w4:
	CMPQ CX, $4
	JL   ai_sse2_bu_scale_tail

	MOVUPS (DI), X0
	MULPS  X3, X0
	MOVUPS X0, (DI)

	ADDQ $16, DI
	SUBQ $4, CX
	JMP  ai_sse2_bu_scale_w4

ai_sse2_bu_scale_tail:
	TESTQ CX, CX
	JZ   ai_sse2_bu_done

ai_sse2_bu_scale_scalar:
	MOVSS (DI), X0
	MULSS X3, X0
	MOVSS X0, (DI)
	ADDQ  $4, DI
	DECQ  CX
	JNZ   ai_sse2_bu_scale_scalar

ai_sse2_bu_done:
	RET

// func FreeEnergyFloat32SSE2Asm(likelihood, posterior, prior *float32, count int) float32
TEXT ·FreeEnergyFloat32SSE2Asm(SB), NOSPLIT, $96-32
	MOVQ likelihood+0(FP), SI
	MOVQ posterior+8(FP), DX
	MOVQ prior+16(FP), R8
	MOVQ count+24(FP), CX

	XORPD X0, X0
	XORPD X1, X1

	MOVSS aiEps<>(SB), X2
	SHUFPS $0, X2, X2

	MOVQ $aiLogC<>(SB), AX
	MOVSS 0(AX), X8
	SHUFPS $0, X8, X8
	MOVSS aiMantMask<>(SB), X4
	SHUFPS $0, X4, X4
	MOVAPS X4, 0(SP)
	MOVSS aiOneBits<>(SB), X4
	SHUFPS $0, X4, X4
	MOVAPS X4, 16(SP)
	MOVSS aiBias127<>(SB), X4
	SHUFPS $0, X4, X4
	MOVAPS X4, 32(SP)

	TESTQ CX, CX
	JZ   ai_sse2_fe_reduce

ai_sse2_fe_w4:
	CMPQ CX, $4
	JL   ai_sse2_fe_tail

	MOVUPS (SI), X3
	MOVUPS (DX), X4
	MOVUPS (R8), X5
	MAXPS X2, X3
	MAXPS X2, X4
	MAXPS X2, X5

	MOVSS 4(AX), X9
	SHUFPS $0, X9, X9
	MOVSS 8(AX), X10
	SHUFPS $0, X10, X10
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

	MOVAPS X3, X0
	CALL ai_sse2_log4(SB)
	MOVAPS X7, 48(SP)

	MOVQ $aiLogC<>(SB), AX
	MOVSS 4(AX), X9
	SHUFPS $0, X9, X9
	MOVSS 8(AX), X10
	SHUFPS $0, X10, X10
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
	MOVAPS X4, X0
	CALL ai_sse2_log4(SB)
	MOVAPS X7, 64(SP)

	MOVQ $aiLogC<>(SB), AX
	MOVSS 4(AX), X9
	SHUFPS $0, X9, X9
	MOVSS 8(AX), X10
	SHUFPS $0, X10, X10
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
	MOVAPS X5, X0
	CALL ai_sse2_log4(SB)
	MOVAPS X7, 80(SP)

	MOVAPS 48(SP), X6
	XORPS X7, X7
	SUBPS X6, X7
	MULPS X4, X7
	CVTPS2PD X7, X3
	ADDPD X3, X0
	MOVAPS X7, X6
	SHUFPS $0xEE, X7, X6
	CVTPS2PD X6, X3
	ADDPD X3, X0

	MOVAPS 64(SP), X6
	MOVAPS 80(SP), X7
	SUBPS X7, X6
	MULPS X4, X6
	CVTPS2PD X6, X3
	ADDPD X3, X1
	MOVAPS X6, X7
	SHUFPS $0xEE, X6, X7
	CVTPS2PD X7, X3
	ADDPD X3, X1

	ADDQ $16, SI
	ADDQ $16, DX
	ADDQ $16, R8
	SUBQ $4, CX
	JMP  ai_sse2_fe_w4

ai_sse2_fe_tail:
	TESTQ CX, CX
	JZ   ai_sse2_fe_reduce

ai_sse2_fe_scalar:
	MOVSS (SI), X3
	MOVSS (DX), X4
	MOVSS (R8), X5
	MAXSS X2, X3
	MAXSS X2, X4
	MAXSS X2, X5

	MOVSS 4(AX), X9
	MOVSS 8(AX), X10
	MOVSS 12(AX), X11
	MOVSS 16(AX), X12
	MOVSS 20(AX), X13
	MOVSS 24(AX), X14
	MOVSS 28(AX), X15
	MOVAPS X3, X0
	CALL ai_sse2_log1(SB)
	MOVSS X7, 48(SP)

	MOVQ $aiLogC<>(SB), AX
	MOVSS 4(AX), X9
	MOVSS 8(AX), X10
	MOVSS 12(AX), X11
	MOVSS 16(AX), X12
	MOVSS 20(AX), X13
	MOVSS 24(AX), X14
	MOVSS 28(AX), X15
	MOVAPS X4, X0
	CALL ai_sse2_log1(SB)
	MOVSS X7, 64(SP)

	MOVQ $aiLogC<>(SB), AX
	MOVSS 4(AX), X9
	MOVSS 8(AX), X10
	MOVSS 12(AX), X11
	MOVSS 16(AX), X12
	MOVSS 20(AX), X13
	MOVSS 24(AX), X14
	MOVSS 28(AX), X15
	MOVAPS X5, X0
	CALL ai_sse2_log1(SB)
	MOVSS X7, 80(SP)

	XORPS X6, X6
	MOVSS 48(SP), X7
	SUBSS X7, X6
	MULSS X4, X6
	CVTSS2SD X6, X6
	ADDSD X6, X0

	MOVSS 64(SP), X6
	MOVSS 80(SP), X7
	SUBSS X7, X6
	MULSS X4, X6
	CVTSS2SD X6, X6
	ADDSD X6, X1

	ADDQ $4, SI
	ADDQ $4, DX
	ADDQ $4, R8
	DECQ CX
	JNZ  ai_sse2_fe_scalar

ai_sse2_fe_reduce:
	ADDPD X1, X0
	MOVAPS X0, X1
	SHUFPD $1, X1, X1
	ADDPD X1, X0
	CVTSD2SS X0, X0
	MOVSS X0, ret+32(FP)
	RET

// ai_sse2_log4 computes natural log of X0 (4-wide), result in X7.
TEXT ai_sse2_log4(SB), NOSPLIT, $0-0
	MOVAPS X0, X1
	PSRLL $23, X1
	MOVAPS 32(SP), X6
	PSUBL X6, X1
	MOVAPS X0, X3
	MOVAPS 0(SP), X6
	PAND X6, X3
	MOVAPS 16(SP), X6
	POR X6, X3
	CVTPL2PS X1, X2
	MOVAPS X3, X0
	SUBPS X9, X0
	MOVAPS X3, X1
	ADDPS X9, X1
	DIVPS X1, X0
	MOVAPS X0, X1
	MULPS X0, X1
	MOVAPS X10, X7
	MULPS X1, X7
	ADDPS X11, X7
	MULPS X1, X7
	ADDPS X12, X7
	MULPS X1, X7
	ADDPS X13, X7
	MULPS X1, X7
	ADDPS X14, X7
	MULPS X1, X7
	ADDPS X9, X7
	MULPS X0, X7
	MULPS X15, X7
	MULPS X8, X2
	ADDPS X2, X7
	RET

// ai_sse2_log1 computes natural log of X0 (scalar), result in X7.
TEXT ai_sse2_log1(SB), NOSPLIT, $0-0
	MOVAPS X0, X1
	PSRLL $23, X1
	MOVSS 32(SP), X6
	PSUBL X6, X1
	MOVAPS X0, X3
	MOVSS 0(SP), X6
	PAND X6, X3
	MOVSS 16(SP), X6
	POR X6, X3
	CVTPL2PS X1, X2
	MOVAPS X3, X0
	SUBSS X9, X0
	MOVAPS X3, X1
	ADDSS X9, X1
	DIVSS X1, X0
	MOVAPS X0, X1
	MULSS X0, X1
	MOVSS X10, X7
	MULSS X1, X7
	ADDSS X11, X7
	MULSS X1, X7
	ADDSS X12, X7
	MULSS X1, X7
	ADDSS X13, X7
	MULSS X1, X7
	ADDSS X14, X7
	MULSS X1, X7
	ADDSS X9, X7
	MULSS X0, X7
	MULSS X15, X7
	MULSS X8, X2
	ADDSS X2, X7
	RET

// func ExpectedFreeEnergyFloat32SSE2Asm(predictedObs, preferredObs, predictedState *float32, obsCount, stateCount int) float32
TEXT ·ExpectedFreeEnergyFloat32SSE2Asm(SB), NOSPLIT, $96-40
	MOVQ predictedObs+0(FP), SI
	MOVQ preferredObs+8(FP), DX
	MOVQ predictedState+16(FP), R8
	MOVQ obsCount+24(FP), CX
	MOVQ stateCount+32(FP), R9

	XORPD X0, X0
	XORPD X1, X1

	MOVSS aiEps<>(SB), X2
	SHUFPS $0, X2, X2

	MOVQ $aiLogC<>(SB), AX
	MOVSS 0(AX), X8
	SHUFPS $0, X8, X8
	MOVSS aiMantMask<>(SB), X4
	SHUFPS $0, X4, X4
	MOVAPS X4, 0(SP)
	MOVSS aiOneBits<>(SB), X4
	SHUFPS $0, X4, X4
	MOVAPS X4, 16(SP)
	MOVSS aiBias127<>(SB), X4
	SHUFPS $0, X4, X4
	MOVAPS X4, 32(SP)

ai_sse2_efe_obs_w4:
	CMPQ CX, $4
	JL   ai_sse2_efe_obs_tail

	MOVUPS (SI), X3
	MOVUPS (DX), X4
	MAXPS X2, X3
	MAXPS X2, X4

	MOVSS 4(AX), X9
	SHUFPS $0, X9, X9
	MOVSS 8(AX), X10
	SHUFPS $0, X10, X10
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
	MOVAPS X3, X0
	CALL ai_sse2_log4(SB)
	MOVAPS X7, 48(SP)

	MOVQ $aiLogC<>(SB), AX
	MOVSS 4(AX), X9
	SHUFPS $0, X9, X9
	MOVSS 8(AX), X10
	SHUFPS $0, X10, X10
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
	MOVAPS X4, X0
	CALL ai_sse2_log4(SB)

	MOVAPS 48(SP), X6
	SUBPS X7, X6
	MULPS X3, X6
	CVTPS2PD X6, X3
	ADDPD X3, X0
	MOVAPS X6, X5
	SHUFPS $0xEE, X6, X5
	CVTPS2PD X5, X3
	ADDPD X3, X0

	ADDQ $16, SI
	ADDQ $16, DX
	SUBQ $4, CX
	JMP  ai_sse2_efe_obs_w4

ai_sse2_efe_obs_tail:
	TESTQ CX, CX
	JZ   ai_sse2_efe_obs_done

ai_sse2_efe_obs_scalar:
	MOVSS (SI), X3
	MOVSS (DX), X4
	MAXSS X2, X3
	MAXSS X2, X4

	MOVQ $aiLogC<>(SB), AX
	MOVSS 4(AX), X9
	MOVSS 8(AX), X10
	MOVSS 12(AX), X11
	MOVSS 16(AX), X12
	MOVSS 20(AX), X13
	MOVSS 24(AX), X14
	MOVSS 28(AX), X15
	MOVAPS X3, X0
	CALL ai_sse2_log1(SB)
	MOVSS X7, 48(SP)

	MOVQ $aiLogC<>(SB), AX
	MOVSS 4(AX), X9
	MOVSS 8(AX), X10
	MOVSS 12(AX), X11
	MOVSS 16(AX), X12
	MOVSS 20(AX), X13
	MOVSS 24(AX), X14
	MOVSS 28(AX), X15
	MOVAPS X4, X0
	CALL ai_sse2_log1(SB)

	MOVSS 48(SP), X6
	SUBSS X7, X6
	MULSS X3, X6
	CVTSS2SD X6, X6
	ADDSD X6, X0

	ADDQ $4, SI
	ADDQ $4, DX
	DECQ CX
	JNZ  ai_sse2_efe_obs_scalar

ai_sse2_efe_obs_done:
	MOVQ predictedState+16(FP), R8
	MOVQ stateCount+32(FP), CX

ai_sse2_efe_state_w4:
	CMPQ CX, $4
	JL   ai_sse2_efe_state_tail

	MOVUPS (R8), X3
	MAXPS X2, X3

	MOVQ $aiLogC<>(SB), AX
	MOVSS 4(AX), X9
	SHUFPS $0, X9, X9
	MOVSS 8(AX), X10
	SHUFPS $0, X10, X10
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
	MOVAPS X3, X0
	CALL ai_sse2_log4(SB)

	XORPS X4, X4
	SUBPS X7, X4
	MULPS X3, X4
	CVTPS2PD X4, X3
	ADDPD X3, X1
	MOVAPS X4, X5
	SHUFPS $0xEE, X4, X5
	CVTPS2PD X5, X3
	ADDPD X3, X1

	ADDQ $16, R8
	SUBQ $4, CX
	JMP  ai_sse2_efe_state_w4

ai_sse2_efe_state_tail:
	TESTQ CX, CX
	JZ   ai_sse2_efe_reduce

ai_sse2_efe_state_scalar:
	MOVSS (R8), X3
	MAXSS X2, X3

	MOVQ $aiLogC<>(SB), AX
	MOVSS 4(AX), X9
	MOVSS 8(AX), X10
	MOVSS 12(AX), X11
	MOVSS 16(AX), X12
	MOVSS 20(AX), X13
	MOVSS 24(AX), X14
	MOVSS 28(AX), X15
	MOVAPS X3, X0
	CALL ai_sse2_log1(SB)

	XORPS X4, X4
	SUBSS X7, X4
	MULSS X3, X4
	CVTSS2SD X4, X4
	ADDSD X4, X1

	ADDQ $4, R8
	DECQ CX
	JNZ  ai_sse2_efe_state_scalar

ai_sse2_efe_reduce:
	ADDPD X1, X0
	MOVAPS X0, X1
	SHUFPD $1, X1, X1
	ADDPD X1, X0
	CVTSD2SS X0, X0
	MOVSS X0, ret+36(FP)
	RET
