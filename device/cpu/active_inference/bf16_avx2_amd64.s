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
