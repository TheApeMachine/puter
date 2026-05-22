// SPDX-License-Identifier: Apache-2.0
// NEON activation steering: dst = base + coefficient * direction.
#include "textflag.h"

#define VFMLA_S4(m, n, d) WORD $(0x4E20CC00 | ((m) << 16) | ((n) << 5) | (d))

// func ActivationSteerFloat32NEONAsm(
//     dst, base, direction *float32,
//     coefficient float32,
//     count int,
// )
TEXT ·ActivationSteerFloat32NEONAsm(SB), NOSPLIT, $0-36
	MOVD dst+0(FP), R0
	MOVD base+8(FP), R1
	MOVD direction+16(FP), R2
	FMOVS coefficient+24(FP), F31
	VDUP V31.S[0], V31.S4
	MOVD count+32(FP), R3

intrp_steer_loop16:
	CMP  $16, R3
	BLT  intrp_steer_loop4

	VLD1 (R1), [V0.S4, V1.S4, V2.S4, V3.S4]
	VLD1 (R2), [V4.S4, V5.S4, V6.S4, V7.S4]
	VFMLA_S4(31, 4, 0)
	VFMLA_S4(31, 5, 1)
	VFMLA_S4(31, 6, 2)
	VFMLA_S4(31, 7, 3)
	VST1 [V0.S4, V1.S4, V2.S4, V3.S4], (R0)

	ADD  $64, R0
	ADD  $64, R1
	ADD  $64, R2
	SUB  $16, R3
	B    intrp_steer_loop16

intrp_steer_loop4:
	CMP  $4, R3
	BLT  intrp_steer_scalar_tail

	VLD1 (R1), [V0.S4]
	VLD1 (R2), [V4.S4]
	VFMLA_S4(31, 4, 0)
	VST1 [V0.S4], (R0)

	ADD  $16, R0
	ADD  $16, R1
	ADD  $16, R2
	SUB  $4, R3
	B    intrp_steer_loop4

intrp_steer_scalar_tail:
	CBZ  R3, intrp_steer_done

intrp_steer_scalar_loop:
	FMOVS (R1), F0
	FMOVS (R2), F1
	FMULS F31, F1, F1
	FADDS F1, F0, F0
	FMOVS F0, (R0)
	ADD  $4, R0
	ADD  $4, R1
	ADD  $4, R2
	SUB  $1, R3
	CBNZ R3, intrp_steer_scalar_loop

intrp_steer_done:
	RET
