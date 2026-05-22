// SPDX-License-Identifier: Apache-2.0
// NEON float32 shape kernels: contiguous copy, where, masked fill.
#include "textflag.h"

#define VBSL_B16(m, n, d) WORD $(0x6E601C00 | ((m) << 16) | ((n) << 5) | (d))
#define VADD_I32(m, n, d) WORD $(0x4EA08400 | ((m) << 16) | ((n) << 5) | (d))
#define VSUB_I32(m, n, d) WORD $(0x6EA08400 | ((m) << 16) | ((n) << 5) | (d))
#define VCGT_I32(m, n, d) WORD $(0x6EA0E800 | ((m) << 16) | ((n) << 5) | (d))

DATA shapeMaskBit4<>+0(SB)/4, $1
DATA shapeMaskBit4<>+4(SB)/4, $2
DATA shapeMaskBit4<>+8(SB)/4, $4
DATA shapeMaskBit4<>+12(SB)/4, $8
GLOBL shapeMaskBit4<>(SB), RODATA|NOPTR, $16

// func CopyContiguousFloat32NEONAsm(dst, src *float32, count int)
TEXT ·CopyContiguousFloat32NEONAsm(SB), NOSPLIT, $0-24
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD count+16(FP), R2

shape_copy_loop16:
	CMP  $16, R2
	BLT  shape_copy_loop4

	VLD1 (R1), [V0.S4, V1.S4, V2.S4, V3.S4]
	VST1 [V0.S4, V1.S4, V2.S4, V3.S4], (R0)

	ADD  $64, R0
	ADD  $64, R1
	SUB  $16, R2
	B    shape_copy_loop16

shape_copy_loop4:
	CMP  $4, R2
	BLT  shape_copy_scalar_tail

	VLD1 (R1), [V0.S4]
	VST1 [V0.S4], (R0)

	ADD  $16, R0
	ADD  $16, R1
	SUB  $4, R2
	B    shape_copy_loop4

shape_copy_scalar_tail:
	CBZ  R2, shape_copy_done

shape_copy_scalar_loop:
	FMOVS (R1), F0
	FMOVS F0, (R0)
	ADD  $4, R0
	ADD  $4, R1
	SUB  $1, R2
	CBNZ R2, shape_copy_scalar_loop

shape_copy_done:
	RET

// Build V8.S4 lane mask (all-ones where bit set) from R10 mask byte and R11 bit offset.
// Clobbers R10-R12, V8-V10, V14-V15.
#define SHAPE_MASK4_R10_R11 \
	LSRW R11, R10, R12; \
	ANDW $0xF, R12; \
	VDUP R12, V10.S4; \
	MOVD $shapeMaskBit4<>(SB), R15; \
	VLD1 (R15), [V14.S4]; \
	VAND V10.B16, V14.B16, V15.B16; \
	VEOR V9.B16, V9.B16, V9.B16; \
	VCGT_I32(9, 15, 8)

// func WhereFloat32NEONAsm(dst, positive, negative *float32, mask *byte, count int)
TEXT ·WhereFloat32NEONAsm(SB), NOSPLIT, $0-40
	MOVD dst+0(FP), R0
	MOVD positive+8(FP), R1
	MOVD negative+16(FP), R2
	MOVD mask+24(FP), R3
	MOVD count+32(FP), R4
	MOVD $0, R5
	MOVD $0, R6

shape_where_loop4:
	CMP  $4, R4
	BLT  shape_where_done

shape_where_load:
	CBNZ R5, shape_where_use

	MOVBU (R3), R6

shape_where_use:
	MOVW  R6, R10
	MOVW  R5, R11
	SHAPE_MASK4_R10_R11

	VLD1 (R1), [V0.S4]
	VLD1 (R2), [V1.S4]
	VBSL  V8.B16, V0.B16, V1.B16
	VST1 [V8.S4], (R0)

	ADD  $16, R0
	ADD  $16, R1
	ADD  $16, R2
	ADD  $4, R5
	CMP  $8, R5
	BLT  shape_where_no_advance

	MOVD  $0, R5
	ADD  $1, R3

shape_where_no_advance:
	SUB  $4, R4
	B    shape_where_loop4

shape_where_done:
	RET

// func MaskedFillFloat32NEONAsm(dst, input *float32, fill float32, mask *byte, count int)
TEXT ·MaskedFillFloat32NEONAsm(SB), NOSPLIT, $0-36
	MOVD dst+0(FP), R0
	MOVD input+8(FP), R1
	FMOVS fill+16(FP), F31
	MOVD mask+24(FP), R3
	MOVD count+32(FP), R4
	MOVD $0, R5
	MOVD $0, R6
	VDUP  V31.S[0], V31.S4

shape_mfill_loop4:
	CMP  $4, R4
	BLT  shape_mfill_done

shape_mfill_load:
	CBNZ R5, shape_mfill_use

	MOVBU (R3), R6

shape_mfill_use:
	MOVW  R6, R10
	MOVW  R5, R11
	SHAPE_MASK4_R10_R11

	VLD1 (R1), [V0.S4]
	VBSL  V8.B16, V31.B16, V0.B16
	VST1 [V8.S4], (R0)

	ADD  $16, R0
	ADD  $16, R1
	ADD  $4, R5
	CMP  $8, R5
	BLT  shape_mfill_no_advance

	MOVD  $0, R5
	ADD  $1, R3

shape_mfill_no_advance:
	SUB  $4, R4
	B    shape_mfill_loop4

shape_mfill_done:
	RET
