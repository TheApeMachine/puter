#include "textflag.h"

#define WIDEN_BF16_8H(baseReg, dstY) \
	VMOVDQU X2, (baseReg); \
	VPMOVZXWD X2, dstY; \
	VPSLLD $16, dstY, dstY; \
	VPSRLDQ $8, X2, X3; \
	VPMOVZXWD X3, Y4; \
	VPSLLD $16, Y4, Y4; \
	VEXTRACTI128 $0, Y4, X4; \
	VINSERTF128 $1, X4, dstY, dstY

#define WIDEN_BF16_4H(baseReg, dstY) \
	VMOVDQU X2, (baseReg); \
	VPMOVZXWD X2, dstY; \
	VPSLLD $16, dstY, dstY

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

#define NARROW_BF16_Y4(dstReg) \
	VPSRLD $16, Y0, Y0; \
	VEXTRACTI128 $0, Y0, X2; \
	MOVL  X2, AX; \
	MOVW  AX, (dstReg); \
	PEXTRD $1, X2, AX; \
	MOVW  AX, 2(dstReg); \
	PEXTRD $2, X2, AX; \
	MOVW  AX, 4(dstReg); \
	PEXTRD $3, X2, AX; \
	MOVW  AX, 6(dstReg)

#define AI_BF16_ACCUM_F64_Y(prodY, accumY) \
	VEXTRACTF128 $0, prodY, X8; \
	VCVTPS2PD X8, Y9; \
	VADDPD accumY, Y9, accumY; \
	VEXTRACTF128 $1, prodY, X8; \
	VCVTPS2PD X8, Y9; \
	VADDPD accumY, Y9, accumY

// func PrecisionWeightBFloat16AVX512Asm(errors, precision, output *uint16, count int)
TEXT ·PrecisionWeightBFloat16AVX512Asm(SB), NOSPLIT, $0-32
	MOVQ errors+0(FP), SI
	MOVQ precision+8(FP), DX
	MOVQ output+16(FP), DI
	MOVQ count+24(FP), CX

	TESTQ CX, CX
	JZ   ai_bf16_avx512_pw_done

ai_bf16_avx512_pw_w8:
	CMPQ CX, $8
	JL   ai_bf16_avx512_pw_w4

	WIDEN_BF16_8H(SI, Y0)
	WIDEN_BF16_8H(DX, Y1)
	VMULPS Y1, Y0, Y0
	NARROW_BF16_Y8(DI)

	ADDQ $16, SI
	ADDQ $16, DX
	ADDQ $16, DI
	SUBQ $8, CX
	JMP  ai_bf16_avx512_pw_w8

ai_bf16_avx512_pw_w4:
	CMPQ CX, $4
	JL   ai_bf16_avx512_pw_tail

	WIDEN_BF16_4H(SI, Y0)
	WIDEN_BF16_4H(DX, Y1)
	VMULPS Y1, Y0, Y0
	NARROW_BF16_Y4(DI)

	ADDQ $8, SI
	ADDQ $8, DX
	ADDQ $8, DI
	SUBQ $4, CX
	JMP  ai_bf16_avx512_pw_w4

ai_bf16_avx512_pw_tail:
	TESTQ CX, CX
	JZ   ai_bf16_avx512_pw_done

ai_bf16_avx512_pw_scalar:
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
	JNZ  ai_bf16_avx512_pw_scalar

ai_bf16_avx512_pw_done:
	RET

// func BeliefUpdateBFloat16AVX512Asm(likelihood, prior, output *uint16, count int)
TEXT ·BeliefUpdateBFloat16AVX512Asm(SB), NOSPLIT, $0-32
	MOVQ likelihood+0(FP), SI
	MOVQ prior+8(FP), DX
	MOVQ output+16(FP), DI
	MOVQ count+24(FP), CX

	TESTQ CX, CX
	JZ   ai_bf16_avx512_bu_done

	VXORPD Y7, Y7, Y7

ai_bf16_avx512_bu_mul_w8:
	CMPQ CX, $8
	JL   ai_bf16_avx512_bu_mul_w4

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
	JMP  ai_bf16_avx512_bu_mul_w8

ai_bf16_avx512_bu_mul_w4:
	CMPQ CX, $4
	JL   ai_bf16_avx512_bu_mul_tail

	WIDEN_BF16_4H(SI, Y1)
	WIDEN_BF16_4H(DX, Y2)
	VMULPS Y2, Y1, Y3
	AI_BF16_ACCUM_F64_Y(Y3, Y7)
	VMOVAPS Y3, Y0
	NARROW_BF16_Y4(DI)

	ADDQ $8, SI
	ADDQ $8, DX
	ADDQ $8, DI
	SUBQ $4, CX
	JMP  ai_bf16_avx512_bu_mul_w4

ai_bf16_avx512_bu_mul_tail:
	TESTQ CX, CX
	JZ   ai_bf16_avx512_bu_reduce

ai_bf16_avx512_bu_mul_scalar:
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
	JNZ  ai_bf16_avx512_bu_mul_scalar

ai_bf16_avx512_bu_reduce:
	VHADDPD Y1, Y7, Y7
	VHADDPD Y1, Y1, Y1
	VEXTRACTF128 $0, Y1, X0
	CVTSD2SS X0, X0

	XORPS X1, X1
	UCOMISS X0, X1
	JZ    ai_bf16_avx512_bu_done

	MOVSS aiOneBits<>(SB), X3
	DIVSS X2, X3
	VBROADCASTSS X3, Y4

	MOVQ output+16(FP), DI
	MOVQ count+24(FP), CX

ai_bf16_avx512_bu_scale_w8:
	CMPQ CX, $8
	JL   ai_bf16_avx512_bu_scale_w4

	WIDEN_BF16_8H(DI, Y0)
	VMULPS Y4, Y0, Y0
	NARROW_BF16_Y8(DI)

	ADDQ $16, DI
	SUBQ $8, CX
	JMP  ai_bf16_avx512_bu_scale_w8

ai_bf16_avx512_bu_scale_w4:
	CMPQ CX, $4
	JL   ai_bf16_avx512_bu_scale_tail

	WIDEN_BF16_4H(DI, Y0)
	VMULPS Y4, Y0, Y0
	NARROW_BF16_Y4(DI)

	ADDQ $8, DI
	SUBQ $4, CX
	JMP  ai_bf16_avx512_bu_scale_w4

ai_bf16_avx512_bu_scale_tail:
	TESTQ CX, CX
	JZ   ai_bf16_avx512_bu_done

ai_bf16_avx512_bu_scale_scalar:
	MOVWLZX (DI), AX
	SHLQ  $16, AX
	VMOVD X2, AX
	VMULSS X3, X2, X2
	VPSRLD $16, X2, X2
	MOVL  X2, AX
	MOVW  AX, (DI)
	ADDQ $2, DI
	DECQ CX
	JNZ  ai_bf16_avx512_bu_scale_scalar

ai_bf16_avx512_bu_done:
	RET
