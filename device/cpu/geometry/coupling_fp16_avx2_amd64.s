#include "textflag.h"
#include "../f16c_fp16_macros.inc"

// func PhaseCouplingFloat16AVX2Asm(dst, left, right *uint16, count int)
TEXT ·PhaseCouplingFloat16AVX2Asm(SB), NOSPLIT, $0-32
	MOVQ destination+0(FP), DI
	MOVQ leftGrowth+8(FP), SI
	MOVQ rightGrowth+16(FP), R8
	MOVQ count+24(FP), CX

pcfp16_avx2_loop:
	TESTQ CX, CX
	JZ    pcfp16_avx2_done

	MOVWLZX (SI), AX
	ANDL  $0x7fff, AX
	VMOVD X2, AX
	VCVTPH2PS X2, X4
	MOVWLZX (R8), DX
	ANDL  $0x7fff, DX
	VMOVD X3, DX
	VCVTPH2PS X3, X6
	VMULSS X6, X4, X4
	VSQRTSS X4, X4, X5
	MOVSS X5, X0
	VCVTPS2PH_X0_X2
	MOVL  X2, AX
	ANDL  $0xffff, AX
	CMPL  AX, $0x211f
	JAE   pcfp16_avx2_compute
	XORL  AX, AX
	MOVW  AX, (DI)
	JMP   pcfp16_avx2_step

pcfp16_avx2_compute:
	VMULSS X6, X4, X7
	VMULSS X5, X5, X5
	VDIVSS X5, X7, X7
	MOVSS X7, X0
	VCVTPS2PH_X0_X2
	MOVL  X2, AX
	MOVW  AX, (DI)

pcfp16_avx2_step:
	ADDQ  $2, SI
	ADDQ  $2, R8
	ADDQ  $2, DI
	DECQ  CX
	JMP   pcfp16_avx2_loop

pcfp16_avx2_done:
	VZEROUPPER
	RET
