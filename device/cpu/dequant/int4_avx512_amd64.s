#include "textflag.h"
#include "x86_int4_macros.inc"

// func DequantInt4AVX512Asm(dst *float32, src *byte, count int, scale float32, zeroPoint int8)
TEXT ·DequantInt4AVX512Asm(SB), NOSPLIT, $0-29
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ count+16(FP), CX
	MOVSS scale+24(FP), X15
	MOVB zeroPoint+28(FP), R8
	SHLQ $56, R8
	SARQ $56, R8

	VBROADCASTSS X15, Y15
	VPBROADCASTD R8, Y14
	VMOVDQU dequantInt4Mask0F<>(SB), X7
	VPXOR X6, X6, X6

	TESTQ CX, CX
	JZ   dequant_i4_done

dequant_i4_w16:
	CMPQ CX, $16
	JL   dequant_i4_w8

	VMOVDQU (SI), X0
	INT4_UNPACK_NIBBLES_ZXBW_X0
	INT4_INTERLEAVE_W16_X3_X4
	INT4_AVX_DEQUAT_Y5_Y6

	ADDQ $8, SI
	ADDQ $64, DI
	SUBQ $16, CX
	JMP  dequant_i4_w16

dequant_i4_w8:
	CMPQ CX, $8
	JL   dequant_i4_scalar_tail

	VMOVD (SI), X0
	INT4_UNPACK_NIBBLES_SSE2_X0
	INT4_INTERLEAVE_W16_X3_X4
	INT4_AVX_DEQUAT_Y5_Y6

	ADDQ $4, SI
	ADDQ $32, DI
	SUBQ $8, CX
	JMP  dequant_i4_w16

dequant_i4_scalar_tail:
	TESTQ CX, CX
	JZ   dequant_i4_done

	MOVQ $0, R10

dequant_i4_scalar_loop:
	MOVB (SI), R9

	CMPQ R10, $0
	JEQ  dequant_i4_take_lo
	SHRQ $4, R9

dequant_i4_take_lo:
	ANDQ $15, R9
	SHLQ $60, R9
	SARQ $60, R9
	SUBQ R8, R9
	IMULQ $4, R9
	CVTSQ2SS R9, X0
	MULSS X15, X0
	MOVSS X0, (DI)

	ADDQ $4, DI
	XORQ $1, R10
	CMPQ R10, $1
	JNE  dequant_i4_next_iter
	ADDQ $1, SI

dequant_i4_next_iter:
	DECQ CX
	JNZ  dequant_i4_scalar_loop

dequant_i4_done:
	RET
