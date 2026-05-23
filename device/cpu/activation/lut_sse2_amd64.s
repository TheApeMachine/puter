// SPDX-License-Identifier: Apache-2.0
// SSE2 uint16 LUT gather via MOVDQU load, PEXTRW/PINSRW gather, MOVDQU store.
#include "textflag.h"

// func ApplyF16LUTSSE2(dst, src *uint16, count int, lut *[65536]uint16)
TEXT ·ApplyF16LUTSSE2(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ count+16(FP), CX
	MOVQ lut+24(FP), BX

	TESTQ CX, CX
	JZ done

sse2_loop4:
	CMPQ CX, $4
	JL sse2_scalar_tail

	MOVDQU (SI), X0

	MOVL X0, AX
	MOVWLZX AX, R8
	LEAQ (BX)(R8*2), R9
	MOVWLZX (R9), R10
	PINSRW $0, R10, X1

	PEXTRW $1, X0, R8
	LEAQ (BX)(R8*2), R9
	MOVWLZX (R9), R10
	PINSRW $1, R10, X1

	PEXTRW $2, X0, R8
	LEAQ (BX)(R8*2), R9
	MOVWLZX (R9), R10
	PINSRW $2, R10, X1

	PEXTRW $3, X0, R8
	LEAQ (BX)(R8*2), R9
	MOVWLZX (R9), R10
	PINSRW $3, R10, X1

	MOVDQU X1, (DI)

	ADDQ $8, SI
	ADDQ $8, DI
	SUBQ $4, CX
	JMP sse2_loop4

sse2_scalar_tail:
	TESTQ CX, CX
	JZ done

sse2_scalar_loop:
	MOVWLZX (SI), R8
	LEAQ (BX)(R8*2), R9
	MOVWLZX (R9), R10
	MOVW R10, (DI)
	ADDQ $2, SI
	ADDQ $2, DI
	DECQ CX
	JNZ sse2_scalar_loop

done:
	RET
