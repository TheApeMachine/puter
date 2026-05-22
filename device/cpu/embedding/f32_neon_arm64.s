// SPDX-License-Identifier: Apache-2.0
// NEON float32 embedding kernels: row copy and row add.
#include "textflag.h"

#define VFADD_S4(m, n, d) WORD $(0x4E20D400 | ((m) << 16) | ((n) << 5) | (d))

// func CopyRowFloat32NEONAsm(dst, src *float32, hidden int)
TEXT ·CopyRowFloat32NEONAsm(SB), NOSPLIT, $0-24
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD hidden+16(FP), R2

emb_copy_loop16:
	CMP  $16, R2
	BLT  emb_copy_loop4

	VLD1 (R1), [V0.S4, V1.S4, V2.S4, V3.S4]
	VST1 [V0.S4, V1.S4, V2.S4, V3.S4], (R0)

	ADD  $64, R0
	ADD  $64, R1
	SUB  $16, R2
	B    emb_copy_loop16

emb_copy_loop4:
	CMP  $4, R2
	BLT  emb_copy_scalar_tail

	VLD1 (R1), [V0.S4]
	VST1 [V0.S4], (R0)

	ADD  $16, R0
	ADD  $16, R1
	SUB  $4, R2
	B    emb_copy_loop4

emb_copy_scalar_tail:
	CBZ  R2, emb_copy_done

emb_copy_scalar_loop:
	FMOVS (R1), F0
	FMOVS F0, (R0)
	ADD  $4, R0
	ADD  $4, R1
	SUB  $1, R2
	CBNZ R2, emb_copy_scalar_loop

emb_copy_done:
	RET

// func AddRowFloat32NEONAsm(dst, src *float32, hidden int)
TEXT ·AddRowFloat32NEONAsm(SB), NOSPLIT, $0-24
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD hidden+16(FP), R2

emb_add_loop16:
	CMP  $16, R2
	BLT  emb_add_loop4

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
	B    emb_add_loop16

emb_add_loop4:
	CMP  $4, R2
	BLT  emb_add_scalar_tail

	VLD1 (R0), [V0.S4]
	VLD1 (R1), [V4.S4]
	VFADD_S4(4, 0, 0)
	VST1 [V0.S4], (R0)

	ADD  $16, R0
	ADD  $16, R1
	SUB  $4, R2
	B    emb_add_loop4

emb_add_scalar_tail:
	CBZ  R2, emb_add_done

emb_add_scalar_loop:
	FMOVS (R0), F0
	FMOVS (R1), F1
	FADDS F1, F0, F0
	FMOVS F0, (R0)
	ADD  $4, R0
	ADD  $4, R1
	SUB  $1, R2
	CBNZ R2, emb_add_scalar_loop

emb_add_done:
	RET
