#include "textflag.h"

#define WIDEN_BF16_8H(baseReg, dstY) \
	VMOVDQU X1, (baseReg); \
	VPXOR X3, X3, X3; \
	VPUNPCKLWD X3, X1, X4; \
	VPUNPCKHWD X3, X1, X5; \
	VPSLLD $16, X4, X4; \
	VPSLLD $16, X5, X5; \
	VINSERTF128 $1, X5, Y4, dstY

#define WIDEN_BF16_4(src, xLo, xHi) \
	VMOVDQU X2, (src); \
	VPXOR  X3, X3, X3; \
	VPUNPCKLWD X3, X2, xLo; \
	VPUNPCKHWD X3, X2, xHi; \
	VPSLLD $16, xLo, xLo; \
	VPSLLD $16, xHi, xHi

#define NARROW_BF16_Y8(dstReg) \
	VPSRLD $16, Y0, Y0; \
	VEXTRACTI128 $0, Y0, X2; \
	MOVL  X2, AX; \
	MOVW  AX, (dstReg); \
	PEXTRD $1, X2, AX; \
	MOVW  AX, 2(dstReg); \
	PEXTRD $2, X2, AX; \
	MOVW  AX, 4(dstReg); \
	PEXTRD $3, X2, AX; \
	MOVW  AX, 6(dstReg); \
	VEXTRACTI128 $1, Y0, X2; \
	MOVL  X2, AX; \
	MOVW  AX, 8(dstReg); \
	PEXTRD $1, X2, AX; \
	MOVW  AX, 10(dstReg); \
	PEXTRD $2, X2, AX; \
	MOVW  AX, 12(dstReg); \
	PEXTRD $3, X2, AX; \
	MOVW  AX, 14(dstReg)

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

#define AI_BF16_ACCUM_F64_Y(prodY, accumY) \
	VEXTRACTF128 $0, prodY, X8; \
	VCVTPS2PD X8, Y9; \
	VADDPD accumY, Y9, accumY; \
	VEXTRACTF128 $1, prodY, X8; \
	VCVTPS2PD X8, Y9; \
	VADDPD accumY, Y9, accumY

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

// func PrecisionWeightBFloat16AVX2Asm(errors, precision, output *uint16, count int)
TEXT ·PrecisionWeightBFloat16AVX2Asm(SB), NOSPLIT, $0-32
	MOVQ errors+0(FP), SI
	MOVQ precision+8(FP), DX
	MOVQ output+16(FP), DI
	MOVQ count+24(FP), CX

	TESTQ CX, CX
	JZ   ai_bf16_avx2_pw_done

ai_bf16_avx2_pw_w8:
	CMPQ CX, $8
	JL   ai_bf16_avx2_pw_w4

	WIDEN_BF16_8H(SI, Y0)
	WIDEN_BF16_8H(DX, Y1)
	VMULPS Y1, Y0, Y0
	NARROW_BF16_Y8(DI)

	ADDQ $16, SI
	ADDQ $16, DX
	ADDQ $16, DI
	SUBQ $8, CX
	JMP  ai_bf16_avx2_pw_w8

ai_bf16_avx2_pw_w4:
	CMPQ CX, $4
	JL   ai_bf16_avx2_pw_tail

	WIDEN_BF16_4(SI, X4, X5)
	WIDEN_BF16_4(DX, X6, X7)
	VMULPS X6, X4, X4
	VMULPS X7, X5, X5
	NARROW_BF16_4(X4, X5, DI)

	ADDQ $8, SI
	ADDQ $8, DX
	ADDQ $8, DI
	SUBQ $4, CX
	JMP  ai_bf16_avx2_pw_w4

ai_bf16_avx2_pw_tail:
	TESTQ CX, CX
	JZ   ai_bf16_avx2_pw_done

ai_bf16_avx2_pw_scalar:
	MOVWLZX (SI), AX
	SHLQ  $16, AX
	VMOVD X2, AX
	MOVWLZX (DX), AX
	SHLQ  $16, AX
	VMOVD X3, AX
	VMULSS X3, X2, X2
	VPSRLD $16, X2, X2
	MOVL  X2, AX
	MOVW  AX, (DI)
	ADDQ $2, SI
	ADDQ $2, DX
	ADDQ $2, DI
	DECQ CX
	JNZ  ai_bf16_avx2_pw_scalar

ai_bf16_avx2_pw_done:
	RET

// func BeliefUpdateBFloat16AVX2Asm(likelihood, prior, output *uint16, count int)
TEXT ·BeliefUpdateBFloat16AVX2Asm(SB), NOSPLIT, $0-32
	MOVQ likelihood+0(FP), SI
	MOVQ prior+8(FP), DX
	MOVQ output+16(FP), DI
	MOVQ count+24(FP), CX

	TESTQ CX, CX
	JZ   ai_bf16_avx2_bu_done

	VXORPD Y7, Y7, Y7

ai_bf16_avx2_bu_mul_w8:
	CMPQ CX, $8
	JL   ai_bf16_avx2_bu_mul_w4

	WIDEN_BF16_8H(SI, Y1)
	WIDEN_BF16_8H(DX, Y2)
	VMULPS Y2, Y1, Y3
	AI_BF16_ACCUM_F64_Y(Y3, Y7)
	VMOVAPS Y3, Y0
	NARROW_BF16_Y8(DI)

	ADDQ $16, SI
	ADDQ $16, DX
	ADDQ $16, DI
	SUBQ $8, CX
	JMP  ai_bf16_avx2_bu_mul_w8

ai_bf16_avx2_bu_mul_w4:
	CMPQ CX, $4
	JL   ai_bf16_avx2_bu_mul_tail

	WIDEN_BF16_4(SI, X4, X5)
	WIDEN_BF16_4(DX, X6, X7)
	VMULPS X6, X4, X4
	VMULPS X7, X5, X5
	AI_BF16_ACCUM_F64_X(X4, X5, X7)
	NARROW_BF16_4(X4, X5, DI)

	ADDQ $8, SI
	ADDQ $8, DX
	ADDQ $8, DI
	SUBQ $4, CX
	JMP  ai_bf16_avx2_bu_mul_w4

ai_bf16_avx2_bu_mul_tail:
	TESTQ CX, CX
	JZ   ai_bf16_avx2_bu_reduce

ai_bf16_avx2_bu_mul_scalar:
	MOVWLZX (SI), AX
	SHLQ  $16, AX
	VMOVD X2, AX
	MOVWLZX (DX), AX
	SHLQ  $16, AX
	VMOVD X3, AX
	VMULSS X3, X2, X2
	MOVSS X2, X1
	CVTSS2SD X1, X1
	ADDSD X1, X7
	VPSRLD $16, X2, X2
	MOVL  X2, AX
	MOVW  AX, (DI)
	ADDQ $2, SI
	ADDQ $2, DX
	ADDQ $2, DI
	DECQ CX
	JNZ  ai_bf16_avx2_bu_mul_scalar

ai_bf16_avx2_bu_reduce:
	VHADDPD Y1, Y7, Y7
	VHADDPD Y1, Y1, Y1
	VEXTRACTF128 $0, Y1, X0
	CVTSD2SS X0, X0

	XORPS X1, X1
	UCOMISS X0, X1
	JZ    ai_bf16_avx2_bu_done

	MOVSS aiOneBits<>(SB), X3
	DIVSS X2, X3
	VBROADCASTSS X3, Y4

	MOVQ output+16(FP), DI
	MOVQ count+24(FP), CX

ai_bf16_avx2_bu_scale_w8:
	CMPQ CX, $8
	JL   ai_bf16_avx2_bu_scale_w4

	WIDEN_BF16_8H(DI, Y0)
	VMULPS Y4, Y0, Y0
	NARROW_BF16_Y8(DI)

	ADDQ $16, DI
	SUBQ $8, CX
	JMP  ai_bf16_avx2_bu_scale_w8

ai_bf16_avx2_bu_scale_w4:
	CMPQ CX, $4
	JL   ai_bf16_avx2_bu_scale_tail

	WIDEN_BF16_4(DI, X4, X5)
	VMULPS X3, X4, X4
	VMULPS X3, X5, X5
	NARROW_BF16_4(X4, X5, DI)

	ADDQ $8, DI
	SUBQ $4, CX
	JMP  ai_bf16_avx2_bu_scale_w4

ai_bf16_avx2_bu_scale_tail:
	TESTQ CX, CX
	JZ   ai_bf16_avx2_bu_done

ai_bf16_avx2_bu_scale_scalar:
	MOVWLZX (DI), AX
	SHLQ  $16, AX
	VMOVD X2, AX
	VMULSS X3, X2, X2
	VPSRLD $16, X2, X2
	MOVL  X2, AX
	MOVW  AX, (DI)
	ADDQ $2, DI
	DECQ CX
	JNZ  ai_bf16_avx2_bu_scale_scalar

ai_bf16_avx2_bu_done:
	RET

#define AI_BF16_AVX2_LOAD_LOG_POLY \
	MOVQ $aiLogC<>(SB), AX; \
	VMOVSS 4(AX), X9; \
	VBROADCASTSS X9, Y9; \
	VMOVSS 8(AX), X10; \
	VBROADCASTSS X10, Y10; \
	VMOVSS 12(AX), X11; \
	VBROADCASTSS X11, Y11; \
	VMOVSS 16(AX), X12; \
	VBROADCASTSS X12, Y12; \
	VMOVSS 20(AX), X13; \
	VBROADCASTSS X13, Y13; \
	VMOVSS 24(AX), X14; \
	VBROADCASTSS X14, Y14; \
	VMOVSS 28(AX), X15; \
	VBROADCASTSS X15, Y15

#define AI_BF16_AVX2_STORE_RESULT \
	VADDPD Y0, Y1, Y0; \
	VHADDPD Y1, Y0, Y0; \
	VEXTRACTF128 $0, Y0, X0; \
	CVTSD2SS X0, X0; \
	VPSRLD $16, X0, X0; \
	MOVL X0, AX; \
	MOVW AX, ret+32(FP)

#define AI_BF16_AVX2_STORE_EFE_RESULT \
	VADDPD Y0, Y1, Y0; \
	VHADDPD Y1, Y0, Y0; \
	VEXTRACTF128 $0, Y0, X0; \
	CVTSD2SS X0, X0; \
	VPSRLD $16, X0, X0; \
	MOVL X0, AX; \
	MOVW AX, ret+40(FP)

// func FreeEnergyBFloat16AVX2Asm(likelihood, posterior, prior *uint16, count int) uint16
TEXT ·FreeEnergyBFloat16AVX2Asm(SB), NOSPLIT, $96-34
	MOVQ likelihood+0(FP), SI
	MOVQ posterior+8(FP), DX
	MOVQ prior+16(FP), R8
	MOVQ count+24(FP), CX

	VXORPD Y0, Y0, Y0
	VXORPD Y1, Y1, Y1

	VBROADCASTSS aiEps<>(SB), Y2

	MOVQ $aiLogC<>(SB), AX
	VMOVSS 0(AX), X8
	VBROADCASTSS X8, Y8
	VPBROADCASTD aiMantMask<>(SB), Y4
	VMOVDQA Y4, 0(SP)
	VPBROADCASTD aiOneBits<>(SB), Y4
	VMOVDQA Y4, 16(SP)
	VPBROADCASTD aiBias127<>(SB), Y4
	VMOVDQA Y4, 32(SP)

	TESTQ CX, CX
	JZ   ai_bf16_avx2_fe_reduce

ai_bf16_avx2_fe_w8:
	CMPQ CX, $8
	JL   ai_bf16_avx2_fe_w4

	WIDEN_BF16_8H(SI, Y3)
	WIDEN_BF16_8H(DX, Y4)
	WIDEN_BF16_8H(R8, Y5)
	VMAXPS Y2, Y3, Y3
	VMAXPS Y2, Y4, Y4
	VMAXPS Y2, Y5, Y5

	AI_BF16_AVX2_LOAD_LOG_POLY
	VMOVAPS Y3, Y0
	CALL ai_avx2_log8(SB)
	VMOVAPS Y7, 48(SP)

	AI_BF16_AVX2_LOAD_LOG_POLY
	VMOVAPS Y4, Y0
	CALL ai_avx2_log8(SB)
	VMOVAPS Y7, 64(SP)

	AI_BF16_AVX2_LOAD_LOG_POLY
	VMOVAPS Y5, Y0
	CALL ai_avx2_log8(SB)
	VMOVAPS Y7, 80(SP)

	VMOVAPS 48(SP), Y6
	VXORPS Y7, Y7, Y7
	VSUBPS Y6, Y7, Y6
	VMULPS Y4, Y6, Y6
	AI_BF16_ACCUM_F64_Y(Y6, Y0)

	VMOVAPS 64(SP), Y6
	VMOVAPS 80(SP), Y7
	VSUBPS Y7, Y6, Y6
	VMULPS Y4, Y6, Y6
	AI_BF16_ACCUM_F64_Y(Y6, Y1)

	ADDQ $16, SI
	ADDQ $16, DX
	ADDQ $16, R8
	SUBQ $8, CX
	JMP  ai_bf16_avx2_fe_w8

ai_bf16_avx2_fe_w4:
	CMPQ CX, $4
	JL   ai_bf16_avx2_fe_tail

	WIDEN_BF16_4(SI, X3, X10)
	WIDEN_BF16_4(DX, X4, X11)
	WIDEN_BF16_4(R8, X5, X12)
	VMAXPS X2, X3, X3
	VMAXPS X2, X4, X4
	VMAXPS X2, X5, X5

	AI_BF16_AVX2_LOAD_LOG_POLY
	VMOVAPS X3, X0
	CALL ai_avx2_log4(SB)
	VMOVAPS X7, 48(SP)

	AI_BF16_AVX2_LOAD_LOG_POLY
	VMOVAPS X4, X0
	CALL ai_avx2_log4(SB)
	VMOVAPS X7, 64(SP)

	AI_BF16_AVX2_LOAD_LOG_POLY
	VMOVAPS X5, X0
	CALL ai_avx2_log4(SB)
	VMOVAPS X7, 80(SP)

	VMOVAPS 48(SP), X6
	VXORPS X7, X7, X7
	VSUBPS X6, X7, X6
	VMULPS X4, X6, X6
	VCVTPS2PD X6, Y3
	VADDPD Y0, Y3, Y0
	MOVAPS X6, X7
	SHUFPS $0xEE, X6, X7
	VCVTPS2PD X7, Y3
	VADDPD Y0, Y3, Y0

	VMOVAPS 64(SP), X6
	VMOVAPS 80(SP), X7
	VSUBPS X7, X6, X6
	VMULPS X4, X6, X6
	VCVTPS2PD X6, Y3
	VADDPD Y1, Y3, Y1
	MOVAPS X6, X7
	SHUFPS $0xEE, X6, X7
	VCVTPS2PD X7, Y3
	VADDPD Y1, Y3, Y1

	ADDQ $8, SI
	ADDQ $8, DX
	ADDQ $8, R8
	SUBQ $4, CX
	JMP  ai_bf16_avx2_fe_w4

ai_bf16_avx2_fe_tail:
	TESTQ CX, CX
	JZ   ai_bf16_avx2_fe_reduce

ai_bf16_avx2_fe_scalar:
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

	AI_BF16_AVX2_LOAD_LOG_POLY
	VMOVAPS X3, X0
	CALL ai_avx2_log1(SB)
	VMOVSS X7, 48(SP)

	AI_BF16_AVX2_LOAD_LOG_POLY
	VMOVAPS X4, X0
	CALL ai_avx2_log1(SB)
	VMOVSS X7, 64(SP)

	AI_BF16_AVX2_LOAD_LOG_POLY
	VMOVAPS X5, X0
	CALL ai_avx2_log1(SB)
	VMOVSS X7, 80(SP)

	VXORPS X6, X6, X6
	VMOVSS 48(SP), X7
	VSUBSS X7, X6, X6
	VMULSS X4, X6, X6
	VCVTSS2SD X6, X6, X6
	VADDSD X6, X0, X0

	VMOVSS 64(SP), X6
	VMOVSS 80(SP), X7
	VSUBSS X7, X6, X6
	VMULSS X4, X6, X6
	VCVTSS2SD X6, X6, X6
	VADDSD X6, X1, X1

	ADDQ $2, SI
	ADDQ $2, DX
	ADDQ $2, R8
	DECQ CX
	JNZ  ai_bf16_avx2_fe_scalar

ai_bf16_avx2_fe_reduce:
	AI_BF16_AVX2_STORE_RESULT
	RET

// func ExpectedFreeEnergyBFloat16AVX2Asm(
//     predictedObs, preferredObs, predictedState *uint16,
//     obsCount, stateCount int,
// ) uint16
TEXT ·ExpectedFreeEnergyBFloat16AVX2Asm(SB), NOSPLIT, $96-42
	MOVQ predictedObs+0(FP), SI
	MOVQ preferredObs+8(FP), DX
	MOVQ predictedState+16(FP), R8
	MOVQ obsCount+24(FP), CX
	MOVQ stateCount+32(FP), R9

	VXORPD Y0, Y0, Y0
	VXORPD Y1, Y1, Y1

	VBROADCASTSS aiEps<>(SB), Y2

	MOVQ $aiLogC<>(SB), AX
	VMOVSS 0(AX), X8
	VBROADCASTSS X8, Y8
	VPBROADCASTD aiMantMask<>(SB), Y4
	VMOVDQA Y4, 0(SP)
	VPBROADCASTD aiOneBits<>(SB), Y4
	VMOVDQA Y4, 16(SP)
	VPBROADCASTD aiBias127<>(SB), Y4
	VMOVDQA Y4, 32(SP)

ai_bf16_avx2_efe_obs_w8:
	CMPQ CX, $8
	JL   ai_bf16_avx2_efe_obs_w4

	WIDEN_BF16_8H(SI, Y3)
	WIDEN_BF16_8H(DX, Y4)
	VMAXPS Y2, Y3, Y3
	VMAXPS Y2, Y4, Y4

	AI_BF16_AVX2_LOAD_LOG_POLY
	VMOVAPS Y3, Y0
	CALL ai_avx2_log8(SB)
	VMOVAPS Y7, 48(SP)

	AI_BF16_AVX2_LOAD_LOG_POLY
	VMOVAPS Y4, Y0
	CALL ai_avx2_log8(SB)

	VMOVAPS 48(SP), Y6
	VSUBPS Y7, Y6, Y6
	VMULPS Y3, Y6, Y6
	AI_BF16_ACCUM_F64_Y(Y6, Y0)

	ADDQ $16, SI
	ADDQ $16, DX
	SUBQ $8, CX
	JMP  ai_bf16_avx2_efe_obs_w8

ai_bf16_avx2_efe_obs_w4:
	CMPQ CX, $4
	JL   ai_bf16_avx2_efe_obs_tail

	WIDEN_BF16_4(SI, X3, X10)
	WIDEN_BF16_4(DX, X4, X11)
	VMAXPS X2, X3, X3
	VMAXPS X2, X4, X4

	AI_BF16_AVX2_LOAD_LOG_POLY
	VMOVAPS X3, X0
	CALL ai_avx2_log4(SB)
	VMOVAPS X7, 48(SP)

	AI_BF16_AVX2_LOAD_LOG_POLY
	VMOVAPS X4, X0
	CALL ai_avx2_log4(SB)

	VMOVAPS 48(SP), X6
	VSUBPS X7, X6, X6
	VMULPS X3, X6, X6
	VCVTPS2PD X6, Y3
	VADDPD Y0, Y3, Y0
	MOVAPS X6, X7
	SHUFPS $0xEE, X6, X7
	VCVTPS2PD X7, Y3
	VADDPD Y0, Y3, Y0

	ADDQ $8, SI
	ADDQ $8, DX
	SUBQ $4, CX
	JMP  ai_bf16_avx2_efe_obs_w4

ai_bf16_avx2_efe_obs_tail:
	TESTQ CX, CX
	JZ   ai_bf16_avx2_efe_obs_done

ai_bf16_avx2_efe_obs_scalar:
	MOVWLZX (SI), AX
	SHLQ  $16, AX
	VMOVD X3, AX
	MOVWLZX (DX), AX
	SHLQ  $16, AX
	VMOVD X4, AX
	MAXSS X2, X3
	MAXSS X2, X4

	AI_BF16_AVX2_LOAD_LOG_POLY
	VMOVAPS X3, X0
	CALL ai_avx2_log1(SB)
	VMOVSS X7, 48(SP)

	AI_BF16_AVX2_LOAD_LOG_POLY
	VMOVAPS X4, X0
	CALL ai_avx2_log1(SB)

	VMOVSS 48(SP), X6
	VSUBSS X7, X6, X6
	VMULSS X3, X6, X6
	VCVTSS2SD X6, X6, X6
	VADDSD X6, X0, X0

	ADDQ $2, SI
	ADDQ $2, DX
	DECQ CX
	JNZ  ai_bf16_avx2_efe_obs_scalar

ai_bf16_avx2_efe_obs_done:
	MOVQ predictedState+16(FP), R8
	MOVQ stateCount+32(FP), CX

ai_bf16_avx2_efe_state_w8:
	CMPQ CX, $8
	JL   ai_bf16_avx2_efe_state_w4

	WIDEN_BF16_8H(R8, Y3)
	VMAXPS Y2, Y3, Y3

	AI_BF16_AVX2_LOAD_LOG_POLY
	VMOVAPS Y3, Y0
	CALL ai_avx2_log8(SB)

	VXORPS Y4, Y4, Y4
	VSUBPS Y7, Y4, Y4
	VMULPS Y3, Y4, Y3
	AI_BF16_ACCUM_F64_Y(Y3, Y1)

	ADDQ $16, R8
	SUBQ $8, CX
	JMP  ai_bf16_avx2_efe_state_w8

ai_bf16_avx2_efe_state_w4:
	CMPQ CX, $4
	JL   ai_bf16_avx2_efe_state_tail

	WIDEN_BF16_4(R8, X3, X10)
	VMAXPS X2, X3, X3

	AI_BF16_AVX2_LOAD_LOG_POLY
	VMOVAPS X3, X0
	CALL ai_avx2_log4(SB)

	VXORPS X4, X4, X4
	VSUBPS X7, X4, X4
	VMULPS X3, X4, X3
	VCVTPS2PD X3, Y4
	VADDPD Y1, Y4, Y1
	MOVAPS X3, X4
	SHUFPS $0xEE, X3, X4
	VCVTPS2PD X4, Y3
	VADDPD Y1, Y3, Y1

	ADDQ $8, R8
	SUBQ $4, CX
	JMP  ai_bf16_avx2_efe_state_w4

ai_bf16_avx2_efe_state_tail:
	TESTQ CX, CX
	JZ   ai_bf16_avx2_efe_reduce

ai_bf16_avx2_efe_state_scalar:
	MOVWLZX (R8), AX
	SHLQ  $16, AX
	VMOVD X3, AX
	MAXSS X2, X3

	AI_BF16_AVX2_LOAD_LOG_POLY
	VMOVAPS X3, X0
	CALL ai_avx2_log1(SB)

	VXORPS X4, X4, X4
	VSUBSS X7, X4, X4
	VMULSS X3, X4, X4
	VCVTSS2SD X4, X4, X4
	VADDSD X4, X1, X1

	ADDQ $2, R8
	DECQ CX
	JNZ  ai_bf16_avx2_efe_state_scalar

ai_bf16_avx2_efe_reduce:
	AI_BF16_AVX2_STORE_EFE_RESULT
	RET
