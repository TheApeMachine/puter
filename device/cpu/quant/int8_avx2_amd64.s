#include "textflag.h"

// func QuantInt8AVX2Asm(dst *int8, src *float32, count int, invScale float32, zeroPoint int32)
TEXT ·QuantInt8AVX2Asm(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ count+16(FP), CX
	MOVSS invScale+24(FP), X15
	MOVL zeroPoint+28(FP), R8

	VBROADCASTSS X15, Y15
	VPBROADCASTD quantClamp32<>(SB), Y14
	MOVL $127, AX
	VPBROADCASTD AX, Y13
	VPBROADCASTD R8, Y12

	TESTQ CX, CX
	JZ   quant_avx2_done

quant_avx2_w8:
	CMPQ CX, $8
	JL   quant_avx2_w4

	VMOVUPS (SI), Y0
	VMULPS  Y0, Y15, Y0
	VROUNDPS $8, Y0, Y0
	VCVTPS2DQ Y0, Y0
	VPADDD  Y0, Y12, Y0
	VPMAXSD Y0, Y14, Y0
	VPMINSD Y0, Y13, Y0
	VEXTRACTI128 $0, Y0, X0
	VEXTRACTI128 $1, Y0, X1
	VPACKSSDW X2, X0, X1
	VPACKSSWB X2, X2, X2
	MOVQ    X2, (DI)

	ADDQ $32, SI
	ADDQ $8, DI
	SUBQ $8, CX
	JMP  quant_avx2_w8

quant_avx2_w4:
	CMPQ CX, $4
	JL   quant_avx2_scalar_tail

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
	JMP  quant_avx2_w4

quant_avx2_scalar_tail:
	TESTQ CX, CX
	JZ   quant_avx2_done

quant_avx2_scalar_loop:
	MOVSS (SI), X0
	MULSS X15, X0
	ROUNDSS $8, X0, X0
	VCVTSS2SI X0, AX
	ADDL R8, AX
	CMPL AX, $127
	JLE  quant_avx2_no_high_sat
	MOVL $127, AX
quant_avx2_no_high_sat:
	CMPL AX, $-128
	JGE  quant_avx2_no_low_sat
	MOVL $-128, AX
quant_avx2_no_low_sat:
	MOVB AX, (DI)
	ADDQ $4, SI
	ADDQ $1, DI
	DECQ CX
	JNZ  quant_avx2_scalar_loop

quant_avx2_done:
	RET
