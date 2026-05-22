// SPDX-License-Identifier: Apache-2.0
// NEON weight graft: in-place weights += injection.
#include "textflag.h"

#define VFADD_S4(m, n, d) WORD $(0x4E20D400 | ((m) << 16) | ((n) << 5) | (d))

// func WeightGraftAddFloat32NEONAsm(weights, injection *float32, count int)
TEXT ·WeightGraftAddFloat32NEONAsm(SB), NOSPLIT, $0-24
	MOVD weights+0(FP), R0
	MOVD injection+8(FP), R1
	MOVD count+16(FP), R2

mdl_graft_loop16:
	CMP  $16, R2
	BLT  mdl_graft_loop4

	VLD1 (R0), [V0.S4, V1.S4, V2.S4, V3.S4]
	VLD1 (R1), [V4.S4, V5.S4, V6.S4, V7.S4]
	VFADD_S4(4, 0, 0)
	VFADD_S4(5, 1, 1)
	VFADD_S4(6, 2, 2)
	VFADD_S4(7, 3, 3)
	VST1 [V0.S4, V1.S4, V2.S4, V3.S4], (R0)

	ADD  $64, R0
	ADD  $64, R1
	SUB  $16, R2
	B    mdl_graft_loop16

mdl_graft_loop4:
	CMP  $4, R2
	BLT  mdl_graft_scalar_tail

	VLD1 (R0), [V0.S4]
	VLD1 (R1), [V4.S4]
	VFADD_S4(4, 0, 0)
	VST1 [V0.S4], (R0)

	ADD  $16, R0
	ADD  $16, R1
	SUB  $4, R2
	B    mdl_graft_loop4

mdl_graft_scalar_tail:
	CBZ  R2, mdl_graft_done

mdl_graft_scalar_loop:
	FMOVS (R0), F0
	FMOVS (R1), F1
	FADDS F1, F0, F0
	FMOVS F0, (R0)
	ADD  $4, R0
	ADD  $4, R1
	SUB  $1, R2
	CBNZ R2, mdl_graft_scalar_loop

mdl_graft_done:
	RET
