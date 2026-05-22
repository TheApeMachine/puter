#include "textflag.h"

#define VCVTPS2PH_X0_X2 WORD $0xC4E3; WORD $0x7D1D; BYTE $0xD0; BYTE $0x00

#define WIDEN_FP16_8H(baseReg, dstY) \
	VMOVDQU X1, (baseReg); \
	VCVTPH2PS X1, dstY; \
	VPSRLDQ $8, X1, X2; \
	VCVTPH2PS X2, Y3; \
	VEXTRACTI128 $0, Y3, X3; \
	VINSERTF128 $1, X3, dstY, dstY

#define WIDEN_FP16_4H(baseReg, dstY) \
	VMOVDQU X1, (baseReg); \
	VCVTPH2PS X1, dstY

#define NARROW_FP16_F32_X0_TO_RET \
	VCVTPS2PH_X0_X2; \
	MOVL  X2, AX; \
	MOVW  AX, ret+24(FP)

// func DotFloat16AVX512Asm(left, right *uint16, count int) uint16
TEXT ·DotFloat16AVX512Asm(SB), NOSPLIT, $0-26
	MOVQ left+0(FP), SI
	MOVQ right+8(FP), DI
	MOVQ count+16(FP), CX

	TESTQ CX, CX
	JZ    dot_fp16_zero

	VXORPS Y0, Y0, Y0

dot_fp16_w8:
	CMPQ CX, $8
	JL   dot_fp16_w4

	WIDEN_FP16_8H(SI, Y1)
	WIDEN_FP16_8H(DI, Y2)
	VMULPS Y2, Y1, Y1
	VADDPS Y1, Y0, Y0

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $8, CX
	JMP  dot_fp16_w8

dot_fp16_w4:
	CMPQ CX, $4
	JL   dot_fp16_reduce

	WIDEN_FP16_4H(SI, Y1)
	WIDEN_FP16_4H(DI, Y2)
	VMULPS Y2, Y1, Y1
	VADDPS Y1, Y0, Y0

	ADDQ $8, SI
	ADDQ $8, DI
	SUBQ $4, CX
	JMP  dot_fp16_w4

dot_fp16_reduce:
	VEXTRACTF128 $1, Y0, X1
	VADDPS       X1, X0, X0
	VHADDPS      X0, X0, X0
	VHADDPS      X0, X0, X0

	TESTQ CX, CX
	JZ    dot_fp16_store

dot_fp16_scalar:
	MOVWLZX (SI), AX
	VMOVD X1, AX
	VCVTPH2PS X1, X1
	MOVWLZX (DI), DX
	VMOVD X2, DX
	VCVTPH2PS X2, X2
	VMULSS X2, X1, X1
	VADDSS X1, X0, X0

	ADDQ $2, SI
	ADDQ $2, DI
	DECQ CX
	JNZ  dot_fp16_scalar

dot_fp16_store:
	NARROW_FP16_F32_X0_TO_RET
	RET

dot_fp16_zero:
	XORPS X0, X0
	NARROW_FP16_F32_X0_TO_RET
	RET
