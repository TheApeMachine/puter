#include "textflag.h"

#define VCVTPS2PH_X0_X2 WORD $0xC4E3; WORD $0x7D1D; BYTE $0xD0; BYTE $0x00

#define WIDEN_FP16_4(src, dst) \
	VMOVDQU X2, (src); \
	VCVTPH2PS X2, dst

#define NARROW_FP16_4(src, dst) \
	MOVAPS src, X0; \
	VCVTPS2PH_X0_X2; \
	VMOVDQU X2, (dst)

#define AI_FP16_ACCUM_F64_X(prodX, accumX) \
	CVTPS2PD prodX, X8; \
	ADDPD X8, accumX; \
	MOVAPS prodX, X9; \
	SHUFPS $0xEE, prodX, X9; \
	CVTPS2PD X9, X8; \
	ADDPD X8, accumX

// func PrecisionWeightFloat16SSE2Asm(errors, precision, output *uint16, count int)
TEXT ·PrecisionWeightFloat16SSE2Asm(SB), NOSPLIT, $0-32
	MOVQ errors+0(FP), SI
	MOVQ precision+8(FP), DX
	MOVQ output+16(FP), DI
	MOVQ count+24(FP), CX

	TESTQ CX, CX
	JZ   ai_fp16_sse2_pw_done

ai_fp16_sse2_pw_w4:
	CMPQ CX, $4
	JL   ai_fp16_sse2_pw_tail

	WIDEN_FP16_4(SI, X4)
	WIDEN_FP16_4(DX, X6)
	MULPS X6, X4
	NARROW_FP16_4(X4, DI)

	ADDQ $8, SI
	ADDQ $8, DX
	ADDQ $8, DI
	SUBQ $4, CX
	JMP  ai_fp16_sse2_pw_w4

ai_fp16_sse2_pw_tail:
	TESTQ CX, CX
	JZ   ai_fp16_sse2_pw_done

ai_fp16_sse2_pw_scalar:
	MOVWLZX (SI), AX
	VMOVD X2, AX
	VCVTPH2PS X2, X2
	MOVWLZX (DX), AX
	VMOVD X3, AX
	VCVTPH2PS X3, X3
	MULSS X3, X2
	VMOVAPS X2, X0
	VCVTPS2PH_X0_X2
	MOVL X2, AX
	MOVW AX, (DI)
	ADDQ $2, SI
	ADDQ $2, DX
	ADDQ $2, DI
	DECQ CX
	JNZ  ai_fp16_sse2_pw_scalar

ai_fp16_sse2_pw_done:
	RET

// func BeliefUpdateFloat16SSE2Asm(likelihood, prior, output *uint16, count int)
TEXT ·BeliefUpdateFloat16SSE2Asm(SB), NOSPLIT, $0-32
	MOVQ likelihood+0(FP), SI
	MOVQ prior+8(FP), DX
	MOVQ output+16(FP), DI
	MOVQ count+24(FP), CX

	TESTQ CX, CX
	JZ   ai_fp16_sse2_bu_done

ai_fp16_sse2_bu_store_w4:
	CMPQ CX, $4
	JL   ai_fp16_sse2_bu_store_tail

	WIDEN_FP16_4(SI, X4)
	WIDEN_FP16_4(DX, X6)
	MULPS X6, X4
	NARROW_FP16_4(X4, DI)

	ADDQ $8, SI
	ADDQ $8, DX
	ADDQ $8, DI
	SUBQ $4, CX
	JMP  ai_fp16_sse2_bu_store_w4

ai_fp16_sse2_bu_store_tail:
	TESTQ CX, CX
	JZ   ai_fp16_sse2_bu_sum

ai_fp16_sse2_bu_store_scalar:
	MOVWLZX (SI), AX
	VMOVD X2, AX
	VCVTPH2PS X2, X2
	MOVWLZX (DX), AX
	VMOVD X3, AX
	VCVTPH2PS X3, X3
	MULSS X3, X2
	VMOVAPS X2, X0
	VCVTPS2PH_X0_X2
	MOVL X2, AX
	MOVW AX, (DI)
	ADDQ $2, SI
	ADDQ $2, DX
	ADDQ $2, DI
	DECQ CX
	JNZ  ai_fp16_sse2_bu_store_scalar

ai_fp16_sse2_bu_sum:
	MOVQ likelihood+0(FP), SI
	MOVQ prior+8(FP), DX
	MOVQ count+24(FP), CX
	XORPD X0, X0

ai_fp16_sse2_bu_sum_loop:
	TESTQ CX, CX
	JZ   ai_fp16_sse2_bu_reduce

	MOVWLZX (SI), AX
	VMOVD X2, AX
	VCVTPH2PS X2, X2
	MOVWLZX (DX), AX
	VMOVD X3, AX
	VCVTPH2PS X3, X3
	MULSS X3, X2
	CVTSS2SD X2, X2
	ADDSD X2, X0
	ADDQ $2, SI
	ADDQ $2, DX
	DECQ CX
	JMP  ai_fp16_sse2_bu_sum_loop

ai_fp16_sse2_bu_reduce:
	MOVAPS X0, X1
	SHUFPD $1, X1, X1
	ADDPD  X1, X0
	CVTSD2SS X0, X0

	XORPS X1, X1
	UCOMISS X0, X1
	JZ    ai_fp16_sse2_bu_done

	MOVSS aiOneBits<>(SB), X3
	DIVSS X2, X3
	SHUFPS $0, X3, X3

	MOVQ output+16(FP), DI
	MOVQ count+24(FP), CX

ai_fp16_sse2_bu_scale_w4:
	CMPQ CX, $4
	JL   ai_fp16_sse2_bu_scale_tail

	WIDEN_FP16_4(DI, X4)
	MULPS X3, X4
	NARROW_FP16_4(X4, DI)

	ADDQ $8, DI
	SUBQ $4, CX
	JMP  ai_fp16_sse2_bu_scale_w4

ai_fp16_sse2_bu_scale_tail:
	TESTQ CX, CX
	JZ   ai_fp16_sse2_bu_done

ai_fp16_sse2_bu_scale_scalar:
	MOVWLZX (DI), AX
	VMOVD X2, AX
	VCVTPH2PS X2, X2
	MULSS X3, X2
	VMOVAPS X2, X0
	VCVTPS2PH_X0_X2
	MOVL X2, AX
	MOVW AX, (DI)
	ADDQ $2, DI
	DECQ CX
	JNZ  ai_fp16_sse2_bu_scale_scalar

ai_fp16_sse2_bu_done:
	RET

#define AI_FP16_SSE2_LOAD_LOG_POLY \
	MOVQ $aiLogC<>(SB), AX; \
	MOVSS 4(AX), X9; \
	SHUFPS $0, X9, X9; \
	MOVSS 8(AX), X10; \
	SHUFPS $0, X10, X10; \
	MOVSS 12(AX), X11; \
	SHUFPS $0, X11, X11; \
	MOVSS 16(AX), X12; \
	SHUFPS $0, X12, X12; \
	MOVSS 20(AX), X13; \
	SHUFPS $0, X13, X13; \
	MOVSS 24(AX), X14; \
	SHUFPS $0, X14, X14; \
	MOVSS 28(AX), X15; \
	SHUFPS $0, X15, X15

#define AI_FP16_SSE2_STORE_RESULT \
	ADDPD X1, X0; \
	MOVAPS X0, X1; \
	SHUFPD $1, X1, X1; \
	ADDPD X1, X0; \
	CVTSD2SS X0, X0; \
	VMOVAPS X0, X0; \
	VCVTPS2PH_X0_X2; \
	MOVL X2, AX; \
	MOVW AX, ret+32(FP)

#define AI_FP16_SSE2_STORE_EFE_RESULT \
	ADDPD X1, X0; \
	MOVAPS X0, X1; \
	SHUFPD $1, X1, X1; \
	ADDPD X1, X0; \
	CVTSD2SS X0, X0; \
	VMOVAPS X0, X0; \
	VCVTPS2PH_X0_X2; \
	MOVL X2, AX; \
	MOVW AX, ret+40(FP)

// func FreeEnergyFloat16SSE2Asm(likelihood, posterior, prior *uint16, count int) uint16
TEXT ·FreeEnergyFloat16SSE2Asm(SB), NOSPLIT, $96-34
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
	JZ   ai_fp16_sse2_fe_reduce

ai_fp16_sse2_fe_w4:
	CMPQ CX, $4
	JL   ai_fp16_sse2_fe_tail

	WIDEN_FP16_4(SI, X3)
	WIDEN_FP16_4(DX, X4)
	WIDEN_FP16_4(R8, X5)
	MAXPS X2, X3
	MAXPS X2, X4
	MAXPS X2, X5

	AI_FP16_SSE2_LOAD_LOG_POLY
	MOVAPS X3, X0
	CALL ai_sse2_log4(SB)
	MOVAPS X7, 48(SP)

	AI_FP16_SSE2_LOAD_LOG_POLY
	MOVAPS X4, X0
	CALL ai_sse2_log4(SB)
	MOVAPS X7, 64(SP)

	AI_FP16_SSE2_LOAD_LOG_POLY
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

	ADDQ $8, SI
	ADDQ $8, DX
	ADDQ $8, R8
	SUBQ $4, CX
	JMP  ai_fp16_sse2_fe_w4

ai_fp16_sse2_fe_tail:
	TESTQ CX, CX
	JZ   ai_fp16_sse2_fe_reduce

ai_fp16_sse2_fe_scalar:
	MOVWLZX (SI), AX
	VMOVD X3, AX
	VCVTPH2PS X3, X3
	MOVWLZX (DX), AX
	VMOVD X4, AX
	VCVTPH2PS X4, X4
	MOVWLZX (R8), AX
	VMOVD X5, AX
	VCVTPH2PS X5, X5
	MAXSS X2, X3
	MAXSS X2, X4
	MAXSS X2, X5

	AI_FP16_SSE2_LOAD_LOG_POLY
	MOVAPS X3, X0
	CALL ai_sse2_log1(SB)
	MOVSS X7, 48(SP)

	AI_FP16_SSE2_LOAD_LOG_POLY
	MOVAPS X4, X0
	CALL ai_sse2_log1(SB)
	MOVSS X7, 64(SP)

	AI_FP16_SSE2_LOAD_LOG_POLY
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

	ADDQ $2, SI
	ADDQ $2, DX
	ADDQ $2, R8
	DECQ CX
	JNZ  ai_fp16_sse2_fe_scalar

ai_fp16_sse2_fe_reduce:
	AI_FP16_SSE2_STORE_RESULT
	RET

// func ExpectedFreeEnergyFloat16SSE2Asm(
//     predictedObs, preferredObs, predictedState *uint16,
//     obsCount, stateCount int,
// ) uint16
TEXT ·ExpectedFreeEnergyFloat16SSE2Asm(SB), NOSPLIT, $96-42
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

ai_fp16_sse2_efe_obs_w4:
	CMPQ CX, $4
	JL   ai_fp16_sse2_efe_obs_tail

	WIDEN_FP16_4(SI, X3)
	WIDEN_FP16_4(DX, X4)
	MAXPS X2, X3
	MAXPS X2, X4

	AI_FP16_SSE2_LOAD_LOG_POLY
	MOVAPS X3, X0
	CALL ai_sse2_log4(SB)
	MOVAPS X7, 48(SP)

	AI_FP16_SSE2_LOAD_LOG_POLY
	MOVAPS X4, X0
	CALL ai_sse2_log4(SB)

	MOVAPS 48(SP), X6
	SUBPS X7, X6
	MULPS X3, X6
	CVTPS2PD X6, X3
	ADDPD X3, X0
	MOVAPS X6, X7
	SHUFPS $0xEE, X6, X7
	CVTPS2PD X7, X3
	ADDPD X3, X0

	ADDQ $8, SI
	ADDQ $8, DX
	SUBQ $4, CX
	JMP  ai_fp16_sse2_efe_obs_w4

ai_fp16_sse2_efe_obs_tail:
	TESTQ CX, CX
	JZ   ai_fp16_sse2_efe_obs_done

ai_fp16_sse2_efe_obs_scalar:
	MOVWLZX (SI), AX
	VMOVD X3, AX
	VCVTPH2PS X3, X3
	MOVWLZX (DX), AX
	VMOVD X4, AX
	VCVTPH2PS X4, X4
	MAXSS X2, X3
	MAXSS X2, X4

	AI_FP16_SSE2_LOAD_LOG_POLY
	MOVAPS X3, X0
	CALL ai_sse2_log1(SB)
	MOVSS X7, 48(SP)

	AI_FP16_SSE2_LOAD_LOG_POLY
	MOVAPS X4, X0
	CALL ai_sse2_log1(SB)

	MOVSS 48(SP), X6
	SUBSS X7, X6
	MULSS X3, X6
	CVTSS2SD X6, X6
	ADDSD X6, X0

	ADDQ $2, SI
	ADDQ $2, DX
	DECQ CX
	JNZ  ai_fp16_sse2_efe_obs_scalar

ai_fp16_sse2_efe_obs_done:
	MOVQ predictedState+16(FP), R8
	MOVQ stateCount+32(FP), CX

ai_fp16_sse2_efe_state_w4:
	CMPQ CX, $4
	JL   ai_fp16_sse2_efe_state_tail

	WIDEN_FP16_4(R8, X3)
	MAXPS X2, X3

	AI_FP16_SSE2_LOAD_LOG_POLY
	MOVAPS X3, X0
	CALL ai_sse2_log4(SB)

	XORPS X4, X4
	SUBPS X7, X4
	MULPS X3, X4
	CVTPS2PD X4, X3
	ADDPD X3, X1
	MOVAPS X4, X3
	SHUFPS $0xEE, X4, X3
	CVTPS2PD X3, X4
	ADDPD X4, X1

	ADDQ $8, R8
	SUBQ $4, CX
	JMP  ai_fp16_sse2_efe_state_w4

ai_fp16_sse2_efe_state_tail:
	TESTQ CX, CX
	JZ   ai_fp16_sse2_efe_reduce

ai_fp16_sse2_efe_state_scalar:
	MOVWLZX (R8), AX
	VMOVD X3, AX
	VCVTPH2PS X3, X3
	MAXSS X2, X3

	AI_FP16_SSE2_LOAD_LOG_POLY
	MOVAPS X3, X0
	CALL ai_sse2_log1(SB)

	XORPS X4, X4
	SUBSS X7, X4
	MULSS X3, X4
	CVTSS2SD X4, X4
	ADDSD X4, X1

	ADDQ $2, R8
	DECQ CX
	JNZ  ai_fp16_sse2_efe_state_scalar

ai_fp16_sse2_efe_reduce:
	AI_FP16_SSE2_STORE_EFE_RESULT
	RET
