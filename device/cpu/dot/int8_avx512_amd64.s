#include "textflag.h"

// func DotInt8AVX512Asm(left, right *int8, count int) int32
TEXT ·DotInt8AVX512Asm(SB), NOSPLIT, $0-28
	MOVQ left+0(FP), SI
	MOVQ right+8(FP), DI
	MOVQ count+16(FP), CX

	TESTQ CX, CX
	JZ    dot_i8_zero

	VPXOR Y4, Y4, Y4
	VPXOR Y5, Y5, Y5
	VPXOR Y6, Y6, Y6
	VPXOR Y7, Y7, Y7

dot_i8_w16:
	CMPQ CX, $16
	JL   dot_i8_w4

	VPMOVSXBD (SI), Y0
	VPMOVSXBD 4(SI), Y1
	VPMOVSXBD 8(SI), Y8
	VPMOVSXBD 12(SI), Y9
	VPMOVSXBD (DI), Y2
	VPMOVSXBD 4(DI), Y3
	VPMOVSXBD 8(DI), Y10
	VPMOVSXBD 12(DI), Y11

	VPMULLD Y0, Y2, Y0
	VPMULLD Y1, Y3, Y1
	VPMULLD Y8, Y10, Y8
	VPMULLD Y9, Y11, Y9
	VPADDD  Y0, Y4, Y4
	VPADDD  Y1, Y5, Y5
	VPADDD  Y8, Y6, Y6
	VPADDD  Y9, Y7, Y7

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $16, CX
	JMP  dot_i8_w16

dot_i8_w4:
	CMPQ CX, $4
	JL   dot_i8_reduce

	VPMOVSXBD (SI), Y0
	VPMOVSXBD (DI), Y1
	VPMULLD Y0, Y1, Y2
	VPADDD  Y2, Y4, Y4

	ADDQ $4, SI
	ADDQ $4, DI
	SUBQ $4, CX
	JMP  dot_i8_w4

dot_i8_reduce:
	VPADDD Y5, Y4, Y4
	VPADDD Y6, Y4, Y4
	VPADDD Y7, Y4, Y4

	VEXTRACTI128 $1, Y4, X5
	VPADDD       X5, X4, X4
	VPSRLDQ      $8, X4, X5
	VPADDD       X5, X4, X4
	VPSRLDQ      $4, X4, X5
	VPADDD       X5, X4, X4
	MOVL         X4, AX

	TESTQ CX, CX
	JZ    dot_i8_store

dot_i8_scalar:
	MOVBQSX (SI), DX
	MOVBQSX (DI), R8
	IMULQ R8, DX
	ADDQ  DX, AX
	ADDQ  $1, SI
	ADDQ  $1, DI
	DECQ  CX
	JNZ   dot_i8_scalar

dot_i8_store:
	MOVL AX, ret+24(FP)
	RET

dot_i8_zero:
	XORL AX, AX
	MOVL AX, ret+24(FP)
	RET
