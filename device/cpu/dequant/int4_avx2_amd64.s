#include "textflag.h"

DATA dequantI4AVX2Mask0F<>+0(SB)/8, $0x0F0F0F0F0F0F0F0F
GLOBL dequantI4AVX2Mask0F<>(SB), RODATA, $8

// func DequantInt4AVX2Asm(dst *float32, src *byte, count int, scale float32, zeroPoint int8)
TEXT ·DequantInt4AVX2Asm(SB), NOSPLIT, $0-29
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ count+16(FP), CX
	MOVSS scale+24(FP), X15
	MOVB zeroPoint+28(FP), R8
	SHLQ $56, R8
	SARQ $56, R8

	VBROADCASTSS X15, Y15
	VPBROADCASTD R8, Y14
	VMOVDQU dequantI4AVX2Mask0F<>(SB), X7
	VPXOR X6, X6, X6

	TESTQ CX, CX
	JZ   dequant_i4_avx2_done

dequant_i4_avx2_w16:
	CMPQ CX, $16
	JL   dequant_i4_avx2_w8

	VMOVDQU (SI), X0

	VPAND X7, X0, X1
	VPMOVZXBW X1, X1
	VPSLLW $12, X1, X1
	VPSRAW $12, X1, X1
	VPMOVZXBW X0, X2
	VPSRLW $4, X2, X2
	VPSLLW $12, X2, X2
	VPSRAW $12, X2, X2

	VPUNPCKLWD X1, X2, X3
	VPUNPCKHWD X1, X2, X4

	VPMOVSXBD X3, Y5
	VPMOVSXBD X4, Y6

	VPSUBD Y5, Y14, Y5
	VPSUBD Y6, Y14, Y6

	VCVTDQ2PS Y5, Y5
	VCVTDQ2PS Y6, Y6

	VMULPS Y5, Y15, Y5
	VMULPS Y6, Y15, Y6

	VMOVUPS Y5, (DI)
	VMOVUPS Y6, 32(DI)

	ADDQ $8, SI
	ADDQ $64, DI
	SUBQ $16, CX
	JMP  dequant_i4_avx2_w16

dequant_i4_avx2_w8:
	CMPQ CX, $8
	JL   dequant_i4_avx2_scalar_tail

	VMOVD (SI), X0

	VPAND X7, X0, X1
	VPUNPCKLBW X1, X6, X1
	VPSLLW $12, X1, X1
	VPSRAW $12, X1, X1
	VPUNPCKLBW X0, X6, X2
	VPSRLW $4, X2, X2
	VPSLLW $12, X2, X2
	VPSRAW $12, X2, X2

	VPUNPCKLWD X1, X2, X3
	VPUNPCKHWD X1, X2, X4

	VPMOVSXBD X3, Y5
	VPMOVSXBD X4, Y6

	VPSUBD Y5, Y14, Y5
	VPSUBD Y6, Y14, Y6

	VCVTDQ2PS Y5, Y5
	VCVTDQ2PS Y6, Y6

	VMULPS Y5, Y15, Y5
	VMULPS Y6, Y15, Y6

	VMOVUPS Y5, (DI)
	VMOVUPS Y6, 32(DI)

	ADDQ $4, SI
	ADDQ $32, DI
	SUBQ $8, CX
	JMP  dequant_i4_avx2_w16

dequant_i4_avx2_scalar_tail:
	TESTQ CX, CX
	JZ   dequant_i4_avx2_done

	MOVQ $0, R10

dequant_i4_avx2_scalar_loop:
	MOVB (SI), R9

	CMPQ R10, $0
	JEQ  dequant_i4_avx2_take_lo
	SHRQ $4, R9

dequant_i4_avx2_take_lo:
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
	JNE  dequant_i4_avx2_next_iter
	ADDQ $1, SI

dequant_i4_avx2_next_iter:
	DECQ CX
	JNZ  dequant_i4_avx2_scalar_loop

dequant_i4_avx2_done:
	RET
