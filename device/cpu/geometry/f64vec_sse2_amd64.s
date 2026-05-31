#include "textflag.h"

// func SumFloat64SSE2Asm(src *float64, count int) float64
TEXT ·SumFloat64SSE2Asm(SB), NOSPLIT, $0-24
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX
	TESTQ CX, CX
	JZ   sum64_sse2_zero

	XORPD X0, X0

sum64_sse2_w2:
	CMPQ CX, $2
	JL   sum64_sse2_tail

	MOVUPD (SI), X1
	ADDPD X1, X0
	ADDQ $16, SI
	SUBQ $2, CX
	JMP  sum64_sse2_w2

sum64_sse2_tail:
	TESTQ CX, CX
	JZ   sum64_sse2_done

sum64_sse2_tail_loop:
	MOVSD (SI), X1
	ADDSD X1, X0
	ADDQ $8, SI
	DECQ CX
	JNZ  sum64_sse2_tail_loop

sum64_sse2_done:
	MOVSD X0, ret+16(FP)
	RET

sum64_sse2_zero:
	XORPS X0, X0
	MOVSD X0, ret+16(FP)
	RET

// func DotFloat64SSE2Asm(left, right *float64, count int) float64
TEXT ·DotFloat64SSE2Asm(SB), NOSPLIT, $0-32
	MOVQ left+0(FP), SI
	MOVQ right+8(FP), DI
	MOVQ count+16(FP), CX
	TESTQ CX, CX
	JZ   dot64_sse2_zero

	XORPD X0, X0

dot64_sse2_w2:
	CMPQ CX, $2
	JL   dot64_sse2_tail

	MOVUPD (SI), X1
	MOVUPD (DI), X2
	MULPD X2, X1
	ADDPD X1, X0
	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $2, CX
	JMP  dot64_sse2_w2

dot64_sse2_tail:
	TESTQ CX, CX
	JZ   dot64_sse2_done

dot64_sse2_tail_loop:
	MOVSD (SI), X1
	MULSD (DI), X1
	ADDSD X1, X0
	ADDQ $8, SI
	ADDQ $8, DI
	DECQ CX
	JNZ  dot64_sse2_tail_loop

dot64_sse2_done:
	MOVSD X0, ret+24(FP)
	RET

dot64_sse2_zero:
	XORPS X0, X0
	MOVSD X0, ret+24(FP)
	RET

// func SumOfSquaresFloat64SSE2Asm(src *float64, count int) float64
TEXT ·SumOfSquaresFloat64SSE2Asm(SB), NOSPLIT, $0-24
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX
	TESTQ CX, CX
	JZ   sumsq64_sse2_zero

	XORPD X0, X0

sumsq64_sse2_w2:
	CMPQ CX, $2
	JL   sumsq64_sse2_tail

	MOVUPD (SI), X1
	MULPD X1, X1
	ADDPD X1, X0
	ADDQ $16, SI
	SUBQ $2, CX
	JMP  sumsq64_sse2_w2

sumsq64_sse2_tail:
	TESTQ CX, CX
	JZ   sumsq64_sse2_done

sumsq64_sse2_tail_loop:
	MOVSD (SI), X1
	MULSD X1, X1
	ADDSD X1, X0
	ADDQ $8, SI
	DECQ CX
	JNZ  sumsq64_sse2_tail_loop

sumsq64_sse2_done:
	MOVSD X0, ret+16(FP)
	RET

sumsq64_sse2_zero:
	XORPS X0, X0
	MOVSD X0, ret+16(FP)
	RET

// func ScaleFloat64SSE2Asm(dst, src *float64, scale float64, count int)
TEXT ·ScaleFloat64SSE2Asm(SB), NOSPLIT, $0-40
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVSD scale+16(FP), X0
	MOVQ count+24(FP), CX
	TESTQ CX, CX
	JZ   scale64_sse2_done

scale64_sse2_w2:
	CMPQ CX, $2
	JL   scale64_sse2_tail

	MOVUPD (SI), X1
	MULPD X0, X1
	MOVUPD X1, (DI)
	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $2, CX
	JMP  scale64_sse2_w2

scale64_sse2_tail:
	TESTQ CX, CX
	JZ   scale64_sse2_done

scale64_sse2_tail_loop:
	MOVSD (SI), X1
	MULSD X0, X1
	MOVSD X1, (DI)
	ADDQ $8, SI
	ADDQ $8, DI
	DECQ CX
	JNZ  scale64_sse2_tail_loop

scale64_sse2_done:
	RET

// func MulFloat64SSE2Asm(dst, left, right *float64, count int)
TEXT ·MulFloat64SSE2Asm(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ left+8(FP), SI
	MOVQ right+16(FP), DX
	MOVQ count+24(FP), CX
	TESTQ CX, CX
	JZ   mul64_sse2_done

mul64_sse2_w2:
	CMPQ CX, $2
	JL   mul64_sse2_tail

	MOVUPD (SI), X0
	MOVUPD (DX), X1
	MULPD X1, X0
	MOVUPD X0, (DI)
	ADDQ $16, SI
	ADDQ $16, DX
	ADDQ $16, DI
	SUBQ $2, CX
	JMP  mul64_sse2_w2

mul64_sse2_tail:
	TESTQ CX, CX
	JZ   mul64_sse2_done

mul64_sse2_tail_loop:
	MOVSD (SI), X0
	MULSD (DX), X0
	MOVSD X0, (DI)
	ADDQ $8, SI
	ADDQ $8, DX
	ADDQ $8, DI
	DECQ CX
	JNZ  mul64_sse2_tail_loop

mul64_sse2_done:
	RET

// func AddFloat64SSE2Asm(dst, left, right *float64, count int)
TEXT ·AddFloat64SSE2Asm(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ left+8(FP), SI
	MOVQ right+16(FP), DX
	MOVQ count+24(FP), CX
	TESTQ CX, CX
	JZ   add64_sse2_done

add64_sse2_w2:
	CMPQ CX, $2
	JL   add64_sse2_tail

	MOVUPD (SI), X0
	ADDPD (DX), X0
	MOVUPD X0, (DI)
	ADDQ $16, SI
	ADDQ $16, DX
	ADDQ $16, DI
	SUBQ $2, CX
	JMP  add64_sse2_w2

add64_sse2_tail:
	TESTQ CX, CX
	JZ   add64_sse2_done

add64_sse2_tail_loop:
	MOVSD (SI), X0
	ADDSD (DX), X0
	MOVSD X0, (DI)
	ADDQ $8, SI
	ADDQ $8, DX
	ADDQ $8, DI
	DECQ CX
	JNZ  add64_sse2_tail_loop

add64_sse2_done:
	RET

// func AddScalarFloat64SSE2Asm(dst, src *float64, offset float64, count int)
TEXT ·AddScalarFloat64SSE2Asm(SB), NOSPLIT, $0-40
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVSD offset+16(FP), X0
	MOVQ count+24(FP), CX
	TESTQ CX, CX
	JZ   adds64_sse2_done

adds64_sse2_w2:
	CMPQ CX, $2
	JL   adds64_sse2_tail

	MOVUPD (SI), X1
	ADDPD X0, X1
	MOVUPD X1, (DI)
	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $2, CX
	JMP  adds64_sse2_w2

adds64_sse2_tail:
	TESTQ CX, CX
	JZ   adds64_sse2_done

adds64_sse2_tail_loop:
	MOVSD (SI), X1
	ADDSD X0, X1
	MOVSD X1, (DI)
	ADDQ $8, SI
	ADDQ $8, DI
	DECQ CX
	JNZ  adds64_sse2_tail_loop

adds64_sse2_done:
	RET

// func SqrtFloat64SSE2Asm(dst, src *float64, count int)
TEXT ·SqrtFloat64SSE2Asm(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ count+16(FP), CX
	TESTQ CX, CX
	JZ   sqrt64_sse2_done

sqrt64_sse2_w2:
	CMPQ CX, $2
	JL   sqrt64_sse2_tail

	MOVUPD (SI), X0
	SQRTPD X0, X0
	MOVUPD X0, (DI)
	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $2, CX
	JMP  sqrt64_sse2_w2

sqrt64_sse2_tail:
	TESTQ CX, CX
	JZ   sqrt64_sse2_done

sqrt64_sse2_tail_loop:
	MOVSD (SI), X0
	SQRTSD X0, X0
	MOVSD X0, (DI)
	ADDQ $8, SI
	ADDQ $8, DI
	DECQ CX
	JNZ  sqrt64_sse2_tail_loop

sqrt64_sse2_done:
	RET

// func MaxFloat64SSE2Asm(src *float64, count int) float64
TEXT ·MaxFloat64SSE2Asm(SB), NOSPLIT, $0-24
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX
	TESTQ CX, CX
	JZ   max64_sse2_zero

	MOVSD (SI), X0
	DECQ CX
	JZ   max64_sse2_done

max64_sse2_loop:
	MOVSD (SI), X1
	MAXSD X1, X0
	ADDQ $8, SI
	DECQ CX
	JNZ  max64_sse2_loop

max64_sse2_done:
	MOVSD X0, ret+16(FP)
	RET

max64_sse2_zero:
	XORPS X0, X0
	MOVSD X0, ret+16(FP)
	RET
