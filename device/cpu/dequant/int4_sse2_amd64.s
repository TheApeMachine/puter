#include "textflag.h"
#include "x86_int4_macros.inc"

// func DequantInt4SSE2Asm(dst *float32, src *byte, count int, scale float32, zeroPoint int8)
TEXT ·DequantInt4SSE2Asm(SB), NOSPLIT, $0-29
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ count+16(FP), CX
	MOVSS scale+24(FP), X15
	SHUFPS $0, X15, X15
	MOVB zeroPoint+28(FP), R8
	SHLQ $56, R8
	SARQ $56, R8
	VMOVD R8, X14
	VPSHUFD $0, X14, X14
	VMOVDQU dequantInt4Mask0F<>(SB), X7
	VPXOR X6, X6, X6

	TESTQ CX, CX
	JZ   dequant_i4_sse2_done

dequant_i4_sse2_w8:
	CMPQ CX, $8
	JL   dequant_i4_sse2_scalar_tail

	VMOVD (SI), X0
	INT4_UNPACK_NIBBLES_SSE2_X0
	INT4_INTERLEAVE_W16_X3_X4
	INT4_SSE2_SXW_TO_I32_X3
	INT4_SSE2_DEQUANT_XMM_X1
	INT4_SSE2_SXW_TO_I32_X4
	INT4_SSE2_DEQUANT_XMM_X1_OFFS(16)

	ADDQ $4, SI
	ADDQ $32, DI
	SUBQ $8, CX
	JMP  dequant_i4_sse2_w8

dequant_i4_sse2_scalar_tail:
	TESTQ CX, CX
	JZ   dequant_i4_sse2_done

	MOVQ $0, R10

dequant_i4_sse2_scalar_loop:
	MOVB (SI), R9

	CMPQ R10, $0
	JEQ  dequant_i4_sse2_take_lo
	SHRQ $4, R9

dequant_i4_sse2_take_lo:
	ANDQ $15, R9
	SHLQ $60, R9
	SARQ $60, R9
	SUBQ R8, R9
	CVTSQ2SS R9, X0
	MULSS X15, X0
	MOVSS X0, (DI)

	ADDQ $4, DI
	XORQ $1, R10
	CMPQ R10, $1
	JNE  dequant_i4_sse2_next_iter
	ADDQ $1, SI

dequant_i4_sse2_next_iter:
	DECQ CX
	JNZ  dequant_i4_sse2_scalar_loop

dequant_i4_sse2_done:
	RET
