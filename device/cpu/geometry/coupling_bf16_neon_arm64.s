#include "textflag.h"
#include "../neon_bf16_macros.inc"

// Per-lane canonical bf16 PhaseCoupling. Threshold uses integer compare on
// RNE-narrowed bf16 bits (0x3c23 = phaseCouplingEpsBF16); no FCMPS.

// func PhaseCouplingBFloat16NEONAsm(dst, left, right *uint16, count int)
TEXT ·PhaseCouplingBFloat16NEONAsm(SB), NOSPLIT, $0-32
	MOVD destination+0(FP), R0
	MOVD leftGrowth+8(FP), R1
	MOVD rightGrowth+16(FP), R2
	MOVD count+24(FP), R3

pcbf16_neon_loop:
	CBZ  R3, pcbf16_neon_done

	MOVHU (R1), R4
	MOVHU (R2), R5
	LSL  $16, R4, R4
	LSL  $16, R5, R5
	FMOVS R4, F0
	FMOVS R5, F1
	FABSS F0, F2
	FABSS F1, F3
	FMULS F2, F3, F4
	FSQRTS F4, F5
	FMOVS F5, R4
	BF16_RNE_BITS_IN_REG(R4)
	MOVD $0x3c23, R8
	CMP  R4, R8
	BLO  pcbf16_neon_compute
	FMOVS ZR, F0
	B    pcbf16_neon_store

pcbf16_neon_compute:
	FMULS F0, F1, F7
	FMULS F5, F5, F8
	FDIVS F8, F7, F0

pcbf16_neon_store:
	BF16_RNE_SCALAR_F0_TO(R4)
	MOVH R4, (R0)
	ADD  $2, R1
	ADD  $2, R2
	ADD  $2, R0
	SUB  $1, R3
	B    pcbf16_neon_loop

pcbf16_neon_done:
	RET
