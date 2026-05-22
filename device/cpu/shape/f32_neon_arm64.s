// SPDX-License-Identifier: Apache-2.0
// NEON float32 shape kernels: contiguous copy, where, masked fill.
#include "textflag.h"

#define VAND_B16(m, n, d) WORD $(0x4E201C00 | ((m) << 16) | ((n) << 5) | (d))
#define VBIC_B16(m, n, d) WORD $(0x6E601C00 | ((m) << 16) | ((n) << 5) | (d))
#define VORR_B16(m, n, d) WORD $(0x6EA01C00 | ((m) << 16) | ((n) << 5) | (d))

DATA shapeMask4Table<>+0(SB)/16, $0x0, $0x0, $0x0, $0x0
DATA shapeMask4Table<>+16(SB)/16, $0xffffffff, $0x0, $0x0, $0x0
DATA shapeMask4Table<>+32(SB)/16, $0x0, $0xffffffff, $0x0, $0x0
DATA shapeMask4Table<>+48(SB)/16, $0xffffffff, $0xffffffff, $0x0, $0x0
DATA shapeMask4Table<>+64(SB)/16, $0x0, $0x0, $0xffffffff, $0x0
DATA shapeMask4Table<>+80(SB)/16, $0xffffffff, $0x0, $0xffffffff, $0x0
DATA shapeMask4Table<>+96(SB)/16, $0x0, $0xffffffff, $0xffffffff, $0x0
DATA shapeMask4Table<>+112(SB)/16, $0xffffffff, $0xffffffff, $0xffffffff, $0x0
DATA shapeMask4Table<>+128(SB)/16, $0x0, $0x0, $0x0, $0xffffffff
DATA shapeMask4Table<>+144(SB)/16, $0xffffffff, $0x0, $0x0, $0xffffffff
DATA shapeMask4Table<>+160(SB)/16, $0x0, $0xffffffff, $0x0, $0xffffffff
DATA shapeMask4Table<>+176(SB)/16, $0xffffffff, $0xffffffff, $0x0, $0xffffffff
DATA shapeMask4Table<>+192(SB)/16, $0x0, $0x0, $0xffffffff, $0xffffffff
DATA shapeMask4Table<>+208(SB)/16, $0xffffffff, $0x0, $0xffffffff, $0xffffffff
DATA shapeMask4Table<>+224(SB)/16, $0x0, $0xffffffff, $0xffffffff, $0xffffffff
DATA shapeMask4Table<>+240(SB)/16, $0xffffffff, $0xffffffff, $0xffffffff, $0xffffffff
GLOBL shapeMask4Table<>(SB), RODATA|NOPTR, $256

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

// Build V13.S4 lane mask from R10 mask byte and R11 bit offset.
#define SHAPE_MASK4_R10_R11 \
	LSRW R11, R10, R12; \
	ANDW $0xF, R12; \
	LSL  $4, R12, R12; \
	MOVD $shapeMask4Table<>(SB), R15; \
	ADD  R12, R15, R15; \
	VLD1 (R15), [V13.S4]

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
	BLT  shape_where_scalar_tail

shape_where_load:
	CBNZ R5, shape_where_use

	MOVBU (R3), R6

shape_where_use:
	MOVW  R6, R10
	MOVW  R5, R11
	SHAPE_MASK4_R10_R11

	VLD1 (R1), [V0.S4]
	VLD1 (R2), [V1.S4]
	VAND_B16(13, 0, 4)
	VBIC_B16(13, 1, 5)
	VORR_B16(5, 4, 0)
	VST1 [V0.S4], (R0)

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

shape_where_scalar_tail:
	CBZ  R4, shape_where_done

shape_where_scalar_loop:
	CBNZ R5, shape_where_scalar_use

	MOVBU (R3), R6

shape_where_scalar_use:
	MOVW  R6, R10
	MOVW  R5, R11
	LSRW  R11, R10, R12
	ANDW  $1, R12
	CBZ   R12, shape_where_scalar_neg

	FMOVS (R1), F0
	B     shape_where_scalar_store

shape_where_scalar_neg:
	FMOVS (R2), F0

shape_where_scalar_store:
	FMOVS F0, (R0)
	ADD  $4, R0
	ADD  $4, R1
	ADD  $4, R2
	ADD  $1, R5
	CMP  $8, R5
	BLT  shape_where_scalar_no_advance

	MOVD  $0, R5
	ADD  $1, R3

shape_where_scalar_no_advance:
	SUB  $1, R4
	CBNZ R4, shape_where_scalar_loop

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
	BLT  shape_mfill_scalar_tail

shape_mfill_load:
	CBNZ R5, shape_mfill_use

	MOVBU (R3), R6

shape_mfill_use:
	MOVW  R6, R10
	MOVW  R5, R11
	SHAPE_MASK4_R10_R11

	VLD1 (R1), [V0.S4]
	VAND_B16(13, 31, 4)
	VBIC_B16(13, 0, 5)
	VORR_B16(5, 4, 0)
	VST1 [V0.S4], (R0)

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

shape_mfill_scalar_tail:
	CBZ  R4, shape_mfill_done

shape_mfill_scalar_loop:
	CBNZ R5, shape_mfill_scalar_use

	MOVBU (R3), R6

shape_mfill_scalar_use:
	MOVW  R6, R10
	MOVW  R5, R11
	LSRW  R11, R10, R12
	ANDW  $1, R12
	CBZ   R12, shape_mfill_scalar_keep

	FMOVS F31, F0
	B     shape_mfill_scalar_store

shape_mfill_scalar_keep:
	FMOVS (R1), F0

shape_mfill_scalar_store:
	FMOVS F0, (R0)
	ADD  $4, R0
	ADD  $4, R1
	ADD  $1, R5
	CMP  $8, R5
	BLT  shape_mfill_scalar_no_advance

	MOVD  $0, R5
	ADD  $1, R3

shape_mfill_scalar_no_advance:
	SUB  $1, R4
	CBNZ R4, shape_mfill_scalar_loop

shape_mfill_done:
	RET
