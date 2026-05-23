#include "textflag.h"

#define WIDEN_BF16_4(src, xLo, xHi) \
	VMOVDQU X2, (src); \
	VPXOR  X3, X3, X3; \
	VPUNPCKLWD X3, X2, xLo; \
	VPUNPCKHWD X3, X2, xHi; \
	VPSLLD $16, xLo, xLo; \
	VPSLLD $16, xHi, xHi

#define NARROW_BF16_4(xLo, xHi, dst) \
	VPSRLD $16, xLo, xLo; \
	VPSRLD $16, xHi, xHi; \
	MOVL  xLo, AX; \
	MOVW  AX, (dst); \
	PEXTRD $1, xLo, AX; \
	MOVW  AX, 2(dst); \
	MOVL  xHi, AX; \
	MOVW  AX, 4(dst); \
	PEXTRD $1, xHi, AX; \
	MOVW  AX, 6(dst)

#define AI_BF16_ACCUM_F64_X(prodLo, prodHi, accumX) \
	CVTPS2PD prodLo, X8; \
	ADDPD X8, accumX; \
	MOVAPS prodLo, X9; \
	SHUFPS $0xEE, prodLo, X9; \
	CVTPS2PD X9, X8; \
	ADDPD X8, accumX; \
	CVTPS2PD prodHi, X8; \
	ADDPD X8, accumX; \
	MOVAPS prodHi, X9; \
	SHUFPS $0xEE, prodHi, X9; \
	CVTPS2PD X9, X8; \
	ADDPD X8, accumX

// func PrecisionWeightBFloat16SSE2Asm(errors, precision, output *uint16, count int)
TEXT ·PrecisionWeightBFloat16SSE2Asm(SB), NOSPLIT, $0-32
	MOVQ errors+0(FP), SI
	MOVQ precision+8(FP), DX
	MOVQ output+16(FP), DI
	MOVQ count+24(FP), CX

	TESTQ CX, CX
	JZ   ai_bf16_sse2_pw_done

ai_bf16_sse2_pw_w4:
	CMPQ CX, $4
	JL   ai_bf16_sse2_pw_tail

	WIDEN_BF16_4(SI, X4, X5)
	WIDEN_BF16_4(DX, X6, X7)
	MULPS X6, X4
	MULPS X7, X5
	NARROW_BF16_4(X4, X5, DI)

	ADDQ $8, SI
	ADDQ $8, DX
	ADDQ $8, DI
	SUBQ $4, CX
	JMP  ai_bf16_sse2_pw_w4

ai_bf16_sse2_pw_tail:
	TESTQ CX, CX
	JZ   ai_bf16_sse2_pw_done

ai_bf16_sse2_pw_scalar:
	MOVWLZX (SI), AX
	SHLQ  $16, AX
	VMOVD X2, AX
	MOVWLZX (DX), AX
	SHLQ  $16, AX
	VMOVD X3, AX
	MULSS X3, X2
	VPSRLD $16, X2, X2
	MOVL  X2, AX
	MOVW  AX, (DI)
	ADDQ $2, SI
	ADDQ $2, DX
	ADDQ $2, DI
	DECQ CX
	JNZ  ai_bf16_sse2_pw_scalar

ai_bf16_sse2_pw_done:
	RET

// func BeliefUpdateBFloat16SSE2Asm(likelihood, prior, output *uint16, count int)
TEXT ·BeliefUpdateBFloat16SSE2Asm(SB), NOSPLIT, $0-32
	MOVQ likelihood+0(FP), SI
	MOVQ prior+8(FP), DX
	MOVQ output+16(FP), DI
	MOVQ count+24(FP), CX

	TESTQ CX, CX
	JZ   ai_bf16_sse2_bu_done

	XORPD X0, X0

ai_bf16_sse2_bu_mul_w4:
	CMPQ CX, $4
	JL   ai_bf16_sse2_bu_mul_tail

	WIDEN_BF16_4(SI, X4, X5)
	WIDEN_BF16_4(DX, X6, X7)
	MULPS X6, X4
	MULPS X7, X5
	AI_BF16_ACCUM_F64_X(X4, X5, X0)
	NARROW_BF16_4(X4, X5, DI)

	ADDQ $8, SI
	ADDQ $8, DX
	ADDQ $8, DI
	SUBQ $4, CX
	JMP  ai_bf16_sse2_bu_mul_w4

ai_bf16_sse2_bu_mul_tail:
	TESTQ CX, CX
	JZ   ai_bf16_sse2_bu_reduce

ai_bf16_sse2_bu_mul_scalar:
	MOVWLZX (SI), AX
	SHLQ  $16, AX
	VMOVD X2, AX
	MOVWLZX (DX), AX
	SHLQ  $16, AX
	VMOVD X3, AX
	MULSS X3, X2
	MOVSS X2, X1
	CVTSS2SD X1, X1
	ADDSD X1, X0
	VPSRLD $16, X2, X2
	MOVL  X2, AX
	MOVW  AX, (DI)
	ADDQ $2, SI
	ADDQ $2, DX
	ADDQ $2, DI
	DECQ CX
	JNZ  ai_bf16_sse2_bu_mul_scalar

ai_bf16_sse2_bu_reduce:
	MOVAPS X0, X1
	SHUFPD $1, X1, X1
	ADDPD  X1, X0
	CVTSD2SS X0, X0

	XORPS X1, X1
	UCOMISS X0, X1
	JZ    ai_bf16_sse2_bu_done

	MOVSS aiOneBits<>(SB), X3
	DIVSS X2, X3
	SHUFPS $0, X3, X3

	MOVQ output+16(FP), DI
	MOVQ count+24(FP), CX

ai_bf16_sse2_bu_scale_w4:
	CMPQ CX, $4
	JL   ai_bf16_sse2_bu_scale_tail

	WIDEN_BF16_4(DI, X4, X5)
	MULPS X3, X4
	MULPS X3, X5
	NARROW_BF16_4(X4, X5, DI)

	ADDQ $8, DI
	SUBQ $4, CX
	JMP  ai_bf16_sse2_bu_scale_w4

ai_bf16_sse2_bu_scale_tail:
	TESTQ CX, CX
	JZ   ai_bf16_sse2_bu_done

ai_bf16_sse2_bu_scale_scalar:
	MOVWLZX (DI), AX
	SHLQ  $16, AX
	VMOVD X2, AX
	MULSS X3, X2
	VPSRLD $16, X2, X2
	MOVL  X2, AX
	MOVW  AX, (DI)
	ADDQ $2, DI
	DECQ CX
	JNZ  ai_bf16_sse2_bu_scale_scalar

ai_bf16_sse2_bu_done:
	RET

#define AI_BF16_SSE2_LOAD_LOG_POLY \
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

#define AI_BF16_SSE2_STORE_RESULT \
	ADDPD X1, X0; \
	MOVAPS X0, X1; \
	SHUFPD $1, X1, X1; \
	ADDPD X1, X0; \
	CVTSD2SS X0, X0; \
	VPSRLD $16, X0, X0; \
	MOVL X0, AX; \
	MOVW AX, ret+32(FP)

#define AI_BF16_SSE2_STORE_EFE_RESULT \
	ADDPD X1, X0; \
	MOVAPS X0, X1; \
	SHUFPD $1, X1, X1; \
	ADDPD X1, X0; \
	CVTSD2SS X0, X0; \
	VPSRLD $16, X0, X0; \
	MOVL X0, AX; \
	MOVW AX, ret+40(FP)

// func FreeEnergyBFloat16SSE2Asm(likelihood, posterior, prior *uint16, count int) uint16
TEXT ·FreeEnergyBFloat16SSE2Asm(SB), NOSPLIT, $96-34
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
	JZ   ai_bf16_sse2_fe_reduce

ai_bf16_sse2_fe_w4:
	CMPQ CX, $4
	JL   ai_bf16_sse2_fe_tail

	WIDEN_BF16_4(SI, X3, X10)
	WIDEN_BF16_4(DX, X4, X11)
	WIDEN_BF16_4(R8, X5, X12)
	MAXPS X2, X3
	MAXPS X2, X4
	MAXPS X2, X5

	AI_BF16_SSE2_LOAD_LOG_POLY
	MOVAPS X3, X0
	CALL ai_sse2_log4(SB)
	MOVAPS X7, 48(SP)

	AI_BF16_SSE2_LOAD_LOG_POLY
	MOVAPS X4, X0
	CALL ai_sse2_log4(SB)
	MOVAPS X7, 64(SP)

	AI_BF16_SSE2_LOAD_LOG_POLY
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
	JMP  ai_bf16_sse2_fe_w4

ai_bf16_sse2_fe_tail:
	TESTQ CX, CX
	JZ   ai_bf16_sse2_fe_reduce

ai_bf16_sse2_fe_scalar:
	MOVWLZX (SI), AX
	SHLQ  $16, AX
	VMOVD X3, AX
	MOVWLZX (DX), AX
	SHLQ  $16, AX
	VMOVD X4, AX
	MOVWLZX (R8), AX
	SHLQ  $16, AX
	VMOVD X5, AX
	MAXSS X2, X3
	MAXSS X2, X4
	MAXSS X2, X5

	AI_BF16_SSE2_LOAD_LOG_POLY
	MOVAPS X3, X0
	CALL ai_sse2_log1(SB)
	MOVSS X7, 48(SP)

	AI_BF16_SSE2_LOAD_LOG_POLY
	MOVAPS X4, X0
	CALL ai_sse2_log1(SB)
	MOVSS X7, 64(SP)

	AI_BF16_SSE2_LOAD_LOG_POLY
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
	JNZ  ai_bf16_sse2_fe_scalar

ai_bf16_sse2_fe_reduce:
	AI_BF16_SSE2_STORE_RESULT
	RET

// func ExpectedFreeEnergyBFloat16SSE2Asm(
//     predictedObs, preferredObs, predictedState *uint16,
//     obsCount, stateCount int,
// ) uint16
TEXT ·ExpectedFreeEnergyBFloat16SSE2Asm(SB), NOSPLIT, $96-42
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

ai_bf16_sse2_efe_obs_w4:
	CMPQ CX, $4
	JL   ai_bf16_sse2_efe_obs_tail

	WIDEN_BF16_4(SI, X3, X10)
	WIDEN_BF16_4(DX, X4, X11)
	MAXPS X2, X3
	MAXPS X2, X4

	AI_BF16_SSE2_LOAD_LOG_POLY
	MOVAPS X3, X0
	CALL ai_sse2_log4(SB)
	MOVAPS X7, 48(SP)

	AI_BF16_SSE2_LOAD_LOG_POLY
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
	JMP  ai_bf16_sse2_efe_obs_w4

ai_bf16_sse2_efe_obs_tail:
	TESTQ CX, CX
	JZ   ai_bf16_sse2_efe_obs_done

ai_bf16_sse2_efe_obs_scalar:
	MOVWLZX (SI), AX
	SHLQ  $16, AX
	VMOVD X3, AX
	MOVWLZX (DX), AX
	SHLQ  $16, AX
	VMOVD X4, AX
	MAXSS X2, X3
	MAXSS X2, X4

	AI_BF16_SSE2_LOAD_LOG_POLY
	MOVAPS X3, X0
	CALL ai_sse2_log1(SB)
	MOVSS X7, 48(SP)

	AI_BF16_SSE2_LOAD_LOG_POLY
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
	JNZ  ai_bf16_sse2_efe_obs_scalar

ai_bf16_sse2_efe_obs_done:
	MOVQ predictedState+16(FP), R8
	MOVQ stateCount+32(FP), CX

ai_bf16_sse2_efe_state_w4:
	CMPQ CX, $4
	JL   ai_bf16_sse2_efe_state_tail

	WIDEN_BF16_4(R8, X3, X10)
	MAXPS X2, X3

	AI_BF16_SSE2_LOAD_LOG_POLY
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
	JMP  ai_bf16_sse2_efe_state_w4

ai_bf16_sse2_efe_state_tail:
	TESTQ CX, CX
	JZ   ai_bf16_sse2_efe_reduce

ai_bf16_sse2_efe_state_scalar:
	MOVWLZX (R8), AX
	SHLQ  $16, AX
	VMOVD X3, AX
	MAXSS X2, X3

	AI_BF16_SSE2_LOAD_LOG_POLY
	MOVAPS X3, X0
	CALL ai_sse2_log1(SB)

	XORPS X4, X4
	SUBSS X7, X4
	MULSS X3, X4
	CVTSS2SD X4, X4
	ADDSD X4, X1

	ADDQ $2, R8
	DECQ CX
	JNZ  ai_bf16_sse2_efe_state_scalar

ai_bf16_sse2_efe_reduce:
	AI_BF16_SSE2_STORE_EFE_RESULT
	RET
