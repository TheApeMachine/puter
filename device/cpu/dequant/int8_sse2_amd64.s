#include "textflag.h"

// func DequantInt8SSE2Asm(dst *float32, src *int8, count int, scale float32, zeroPoint int16)
TEXT ·DequantInt8SSE2Asm(SB), NOSPLIT, $0-30
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ count+16(FP), CX
	MOVSS scale+24(FP), X15
	SHUFPS $0, X15, X15
	MOVW zeroPoint+28(FP), R8
	SHLQ $48, R8
	SARQ $48, R8
	VMOVD R8, X14
	VPSHUFD $0, X14, X14

	TESTQ CX, CX
	JZ   dequant_i8_sse2_done

dequant_i8_sse2_w4:
	CMPQ CX, $4
	JL   dequant_i8_sse2_scalar_tail

	VMOVD (SI), X0
	VPUNPCKLBW X0, X0, X0
	VPSRAW $8, X0, X0
	MOVAPS X0, X2
	VPUNPCKLWD X0, X0, X0
	VPSRAD $16, X0, X0
	VPUNPCKHWD X2, X2, X2
	VPSRAD $16, X2, X2
	MOVLHPS X2, X0
	VPSUBD X14, X0, X0
	VCVTDQ2PS X0, X0
	VMULPS X15, X0, X0
	MOVUPS X0, (DI)

	ADDQ $4, SI
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  dequant_i8_sse2_w4

dequant_i8_sse2_scalar_tail:
	TESTQ CX, CX
	JZ   dequant_i8_sse2_done

dequant_i8_sse2_scalar_loop:
	MOVB (SI), R9
	SHLQ $56, R9
	SARQ $56, R9
	SUBQ R8, R9
	CVTSQ2SS R9, X0
	MULSS X15, X0
	MOVSS X0, (DI)

	ADDQ $1, SI
	ADDQ $4, DI
	DECQ CX
	JNZ  dequant_i8_sse2_scalar_loop

dequant_i8_sse2_done:
	RET
