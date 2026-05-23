// SPDX-License-Identifier: Apache-2.0
// NEON uint16 LUT gather: 8-wide unrolled table gather (65536-entry LUT has no
// native gather on AArch64; lane results assembled via MOVH stores).
#include "textflag.h"

// func ApplyF16LUTNEON(dst, src *uint16, count int, lut *[65536]uint16)
TEXT ·ApplyF16LUTNEON(SB), 4, $0-32
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD count+16(FP), R3
	MOVD lut+24(FP), R2

	CBZ R3, done

loop8:
	CMP $8, R3
	BLT scalar_tail

	MOVHU 0(R1), R4
	LSL $1, R4, R6
	ADD R2, R6, R6
	MOVHU (R6), R5
	MOVH R5, 0(R0)

	MOVHU 2(R1), R4
	LSL $1, R4, R6
	ADD R2, R6, R6
	MOVHU (R6), R5
	MOVH R5, 2(R0)

	MOVHU 4(R1), R4
	LSL $1, R4, R6
	ADD R2, R6, R6
	MOVHU (R6), R5
	MOVH R5, 4(R0)

	MOVHU 6(R1), R4
	LSL $1, R4, R6
	ADD R2, R6, R6
	MOVHU (R6), R5
	MOVH R5, 6(R0)

	MOVHU 8(R1), R4
	LSL $1, R4, R6
	ADD R2, R6, R6
	MOVHU (R6), R5
	MOVH R5, 8(R0)

	MOVHU 10(R1), R4
	LSL $1, R4, R6
	ADD R2, R6, R6
	MOVHU (R6), R5
	MOVH R5, 10(R0)

	MOVHU 12(R1), R4
	LSL $1, R4, R6
	ADD R2, R6, R6
	MOVHU (R6), R5
	MOVH R5, 12(R0)

	MOVHU 14(R1), R4
	LSL $1, R4, R6
	ADD R2, R6, R6
	MOVHU (R6), R5
	MOVH R5, 14(R0)

	ADD $16, R1
	ADD $16, R0
	SUB $8, R3
	B loop8

scalar_tail:
	CBZ R3, done

scalar_loop:
	MOVHU (R1), R4
	LSL $1, R4, R6
	ADD R2, R6, R6
	MOVHU (R6), R5
	MOVH R5, (R0)
	ADD $2, R1
	ADD $2, R0
	SUB $1, R3
	CBNZ R3, scalar_loop

done:
	RET
