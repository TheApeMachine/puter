#include "textflag.h"

#define SIGN_EXTEND_4I8_TO_4I16(reg) \
	PXOR X1, X1; \
	PCMPGTB reg, X1; \
	PUNPCKLBW X1, reg

// func DotInt8SSE2Asm(left, right *int8, count int) int32
TEXT ·DotInt8SSE2Asm(SB), NOSPLIT, $0-28
	MOVQ left+0(FP), SI
	MOVQ right+8(FP), DI
	MOVQ count+16(FP), CX

	TESTQ CX, CX
	JZ    dot_i8_sse2_zero

	XORL AX, AX

dot_i8_sse2_w4:
	CMPQ CX, $4
	JL   dot_i8_sse2_tail

	MOVL (SI), R8
	MOVL (DI), R9
	MOVL R8, X0
	MOVL R9, X2

	SIGN_EXTEND_4I8_TO_4I16(X0)
	SIGN_EXTEND_4I8_TO_4I16(X2)

	VPMADDWD X2, X0, X0
	MOVD  X0, DX
	ADDL  DX, AX

	ADDQ $4, SI
	ADDQ $4, DI
	SUBQ $4, CX
	JMP  dot_i8_sse2_w4

dot_i8_sse2_tail:
	TESTQ CX, CX
	JZ    dot_i8_sse2_store

	MOVBQSX (SI), DX
	MOVBQSX (DI), R8
	IMULQ R8, DX
	ADDQ  DX, AX
	ADDQ  $1, SI
	ADDQ  $1, DI
	DECQ  CX
	JMP   dot_i8_sse2_tail

dot_i8_sse2_store:
	MOVL AX, ret+24(FP)
	RET

dot_i8_sse2_zero:
	XORL AX, AX
	MOVL AX, ret+24(FP)
	RET
