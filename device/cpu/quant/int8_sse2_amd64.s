#include "textflag.h"

// func QuantInt8SSE2Asm(dst *int8, src *float32, count int, invScale float32, zeroPoint int32)
TEXT ·QuantInt8SSE2Asm(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ count+16(FP), CX
	MOVSS invScale+24(FP), X15
	MOVL zeroPoint+28(FP), R8

	VBROADCASTSS X15, X15
	VPBROADCASTD quantClamp32<>(SB), X14
	MOVL $127, AX
	VPBROADCASTD AX, X13
	VPBROADCASTD R8, X12

	TESTQ CX, CX
	JZ   quant_sse2_done

quant_sse2_w4:
	CMPQ CX, $4
	JL   quant_sse2_scalar_tail

	VMOVUPS (SI), X0
	VMULPS  X0, X15, X0
	VROUNDPS $8, X0, X0
	VCVTPS2DQ X0, X0
	VPADDD  X0, X12, X0
	VPMAXSD X0, X14, X0
	VPMINSD X0, X13, X0
	VPACKSSDW X0, X0, X0
	VPACKSSWB X0, X0, X0
	MOVL    X0, (DI)

	ADDQ $16, SI
	ADDQ $4, DI
	SUBQ $4, CX
	JMP  quant_sse2_w4

quant_sse2_scalar_tail:
	TESTQ CX, CX
	JZ   quant_sse2_done

quant_sse2_scalar_loop:
	MOVSS (SI), X0
	MULSS X15, X0
	ROUNDSS $8, X0, X0
	VCVTSS2SI X0, AX
	ADDL R8, AX
	CMPL AX, $127
	JLE  quant_sse2_no_high_sat
	MOVL $127, AX
quant_sse2_no_high_sat:
	CMPL AX, $-128
	JGE  quant_sse2_no_low_sat
	MOVL $-128, AX
quant_sse2_no_low_sat:
	MOVB AX, (DI)
	ADDQ $4, SI
	ADDQ $1, DI
	DECQ CX
	JNZ  quant_sse2_scalar_loop

quant_sse2_done:
	RET
