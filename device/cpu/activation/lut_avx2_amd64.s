// SPDX-License-Identifier: Apache-2.0
// AVX2 uint16 LUT gather: two SSE2-style 4-lane integer gathers per 8 elements.
// No float32 load, mask, or narrow — uint16 index → uint16 table load → uint16 store.
#include "textflag.h"

// gather4 loads four uint16 indices from (SI), gathers lut[·], stores four uint16 at (DI).
#define LUT_GATHER4(SI, DI, BX, AX, R8, R9, R10, X0, X1) \
	MOVDQU (SI), X0; \
	\
	MOVL X0, AX; \
	MOVWLZX AX, R8; \
	LEAQ (BX)(R8*2), R9; \
	MOVWLZX (R9), R10; \
	PINSRW $0, R10, X1; \
	\
	PEXTRW $1, X0, R8; \
	LEAQ (BX)(R8*2), R9; \
	MOVWLZX (R9), R10; \
	PINSRW $1, R10, X1; \
	\
	PEXTRW $2, X0, R8; \
	LEAQ (BX)(R8*2), R9; \
	MOVWLZX (R9), R10; \
	PINSRW $2, R10, X1; \
	\
	PEXTRW $3, X0, R8; \
	LEAQ (BX)(R8*2), R9; \
	MOVWLZX (R9), R10; \
	PINSRW $3, R10, X1; \
	\
	MOVDQU X1, (DI)

// func ApplyF16LUTAVX2(dst, src *uint16, count int, lut *[65536]uint16)
TEXT ·ApplyF16LUTAVX2(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ count+16(FP), CX
	MOVQ lut+24(FP), BX

	TESTQ CX, CX
	JZ done

avx2_loop8:
	CMPQ CX, $8
	JL avx2_loop4

	LUT_GATHER4(SI, DI, BX, AX, R8, R9, R10, X0, X1)
	LUT_GATHER4(8(SI), 8(DI), BX, AX, R8, R9, R10, X0, X1)

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $8, CX
	JMP avx2_loop8

avx2_loop4:
	CMPQ CX, $4
	JL avx2_scalar_tail

	LUT_GATHER4(SI, DI, BX, AX, R8, R9, R10, X0, X1)

	ADDQ $8, SI
	ADDQ $8, DI
	SUBQ $4, CX
	JMP avx2_loop4

avx2_scalar_tail:
	TESTQ CX, CX
	JZ done

avx2_scalar_loop:
	MOVWLZX (SI), R8
	LEAQ (BX)(R8*2), R9
	MOVWLZX (R9), R10
	MOVW R10, (DI)
	ADDQ $2, SI
	ADDQ $2, DI
	DECQ CX
	JNZ avx2_scalar_loop

done:
	RET
