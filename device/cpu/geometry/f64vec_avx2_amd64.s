#include "textflag.h"

// func SumFloat64AVX2Asm(src *float64, count int) float64
TEXT ·SumFloat64AVX2Asm(SB), NOSPLIT, $0-24
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX
	TESTQ CX, CX
	JZ   sum64_avx2_zero

	VXORPD Y0, Y0, Y0

sum64_avx2_w4:
	CMPQ CX, $4
	JL   sum64_avx2_w2

	VMOVUPD (SI), Y1
	VADDPD  Y1, Y0, Y0
	ADDQ $32, SI
	SUBQ $4, CX
	JMP  sum64_avx2_w4

sum64_avx2_w2:
	CMPQ CX, $2
	JL   sum64_avx2_reduce

	VMOVUPD (SI), X1
	VADDPD  X1, X0, X0
	ADDQ $16, SI
	SUBQ $2, CX
	JMP  sum64_avx2_w2

sum64_avx2_reduce:
	VEXTRACTF128 $1, Y0, X1
	VADDPD       X1, X0, X0
	VHADDPD      X0, X0, X0

	TESTQ CX, CX
	JZ   sum64_avx2_done

sum64_avx2_tail:
	MOVSD (SI), X1
	VADDSD X1, X0, X0
	ADDQ $8, SI
	DECQ CX
	JNZ  sum64_avx2_tail

sum64_avx2_done:
	VZEROUPPER
	MOVSD X0, ret+16(FP)
	RET

sum64_avx2_zero:
	XORPS X0, X0
	MOVSD X0, ret+16(FP)
	RET

// func SumOfSquaresFloat64AVX2Asm(src *float64, count int) float64
TEXT ·SumOfSquaresFloat64AVX2Asm(SB), NOSPLIT, $0-24
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX
	TESTQ CX, CX
	JZ   sumsq64_avx2_zero

	VXORPD Y0, Y0, Y0

sumsq64_avx2_w4:
	CMPQ CX, $4
	JL   sumsq64_avx2_w2

	VMOVUPD (SI), Y1
	VMULPD  Y1, Y1, Y1
	VADDPD  Y1, Y0, Y0
	ADDQ $32, SI
	SUBQ $4, CX
	JMP  sumsq64_avx2_w4

sumsq64_avx2_w2:
	CMPQ CX, $2
	JL   sumsq64_avx2_reduce

	VMOVUPD (SI), X1
	VMULPD  X1, X1, X1
	VADDPD  X1, X0, X0
	ADDQ $16, SI
	SUBQ $2, CX
	JMP  sumsq64_avx2_w2

sumsq64_avx2_reduce:
	VEXTRACTF128 $1, Y0, X1
	VADDPD       X1, X0, X0
	VHADDPD      X0, X0, X0

	TESTQ CX, CX
	JZ   sumsq64_avx2_done

sumsq64_avx2_tail:
	MOVSD (SI), X1
	VMULSD X1, X1, X1
	VADDSD X1, X0, X0
	ADDQ $8, SI
	DECQ CX
	JNZ  sumsq64_avx2_tail

sumsq64_avx2_done:
	VZEROUPPER
	MOVSD X0, ret+16(FP)
	RET

sumsq64_avx2_zero:
	XORPS X0, X0
	MOVSD X0, ret+16(FP)
	RET

// func DotFloat64AVX2Asm(left, right *float64, count int) float64
TEXT ·DotFloat64AVX2Asm(SB), NOSPLIT, $0-32
	MOVQ left+0(FP), SI
	MOVQ right+8(FP), DI
	MOVQ count+16(FP), CX
	TESTQ CX, CX
	JZ   dot64_avx2_zero

	VXORPD Y0, Y0, Y0

dot64_avx2_w4:
	CMPQ CX, $4
	JL   dot64_avx2_w2

	VMOVUPD (SI), Y1
	VMOVUPD (DI), Y2
	VMULPD  Y2, Y1, Y1
	VADDPD  Y1, Y0, Y0
	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $4, CX
	JMP  dot64_avx2_w4

dot64_avx2_w2:
	CMPQ CX, $2
	JL   dot64_avx2_reduce

	VMOVUPD (SI), X1
	VMOVUPD (DI), X2
	VMULPD  X2, X1, X1
	VADDPD  X1, X0, X0
	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $2, CX
	JMP  dot64_avx2_w2

dot64_avx2_reduce:
	VEXTRACTF128 $1, Y0, X1
	VADDPD       X1, X0, X0
	VHADDPD      X0, X0, X0

	TESTQ CX, CX
	JZ   dot64_avx2_done

dot64_avx2_tail:
	MOVSD (SI), X1
	VMULSD (DI), X1, X1
	VADDSD X1, X0, X0
	ADDQ $8, SI
	ADDQ $8, DI
	DECQ CX
	JNZ  dot64_avx2_tail

dot64_avx2_done:
	VZEROUPPER
	MOVSD X0, ret+24(FP)
	RET

dot64_avx2_zero:
	XORPS X0, X0
	MOVSD X0, ret+24(FP)
	RET

// func ScaleFloat64AVX2Asm(dst, src *float64, scale float64, count int)
TEXT ·ScaleFloat64AVX2Asm(SB), NOSPLIT, $0-40
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVSD scale+16(FP), X0
	MOVQ count+24(FP), CX
	TESTQ CX, CX
	JZ   scale64_avx2_done

	VBROADCASTSD X0, Y0

scale64_avx2_w4:
	CMPQ CX, $4
	JL   scale64_avx2_w2

	VMOVUPD (SI), Y1
	VMULPD  Y0, Y1, Y1
	VMOVUPD Y1, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $4, CX
	JMP  scale64_avx2_w4

scale64_avx2_w2:
	CMPQ CX, $2
	JL   scale64_avx2_tail

	VMOVUPD (SI), X1
	VMULPD  X0, X1, X1
	VMOVUPD X1, (DI)
	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $2, CX
	JMP  scale64_avx2_w2

scale64_avx2_tail:
	TESTQ CX, CX
	JZ   scale64_avx2_done

scale64_avx2_tail_loop:
	MOVSD (SI), X1
	VMULSD X0, X1, X1
	MOVSD X1, (DI)
	ADDQ $8, SI
	ADDQ $8, DI
	DECQ CX
	JNZ  scale64_avx2_tail_loop

scale64_avx2_done:
	VZEROUPPER
	RET

// func MulFloat64AVX2Asm(dst, left, right *float64, count int)
TEXT ·MulFloat64AVX2Asm(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ left+8(FP), SI
	MOVQ right+16(FP), DX
	MOVQ count+24(FP), CX
	TESTQ CX, CX
	JZ   mul64_avx2_done

mul64_avx2_w4:
	CMPQ CX, $4
	JL   mul64_avx2_w2

	VMOVUPD (SI), Y0
	VMOVUPD (DX), Y1
	VMULPD  Y1, Y0, Y0
	VMOVUPD Y0, (DI)
	ADDQ $32, SI
	ADDQ $32, DX
	ADDQ $32, DI
	SUBQ $4, CX
	JMP  mul64_avx2_w4

mul64_avx2_w2:
	CMPQ CX, $2
	JL   mul64_avx2_tail

	VMOVUPD (SI), X0
	VMOVUPD (DX), X1
	VMULPD  X1, X0, X0
	VMOVUPD X0, (DI)
	ADDQ $16, SI
	ADDQ $16, DX
	ADDQ $16, DI
	SUBQ $2, CX
	JMP  mul64_avx2_w2

mul64_avx2_tail:
	TESTQ CX, CX
	JZ   mul64_avx2_done

mul64_avx2_tail_loop:
	MOVSD (SI), X0
	VMULSD (DX), X0, X0
	MOVSD X0, (DI)
	ADDQ $8, SI
	ADDQ $8, DX
	ADDQ $8, DI
	DECQ CX
	JNZ  mul64_avx2_tail_loop

mul64_avx2_done:
	VZEROUPPER
	RET

// func AddFloat64AVX2Asm(dst, left, right *float64, count int)
TEXT ·AddFloat64AVX2Asm(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ left+8(FP), SI
	MOVQ right+16(FP), DX
	MOVQ count+24(FP), CX
	TESTQ CX, CX
	JZ   add64_avx2_done

add64_avx2_w4:
	CMPQ CX, $4
	JL   add64_avx2_w2

	VMOVUPD (SI), Y0
	VADDPD  (DX), Y0, Y0
	VMOVUPD Y0, (DI)
	ADDQ $32, SI
	ADDQ $32, DX
	ADDQ $32, DI
	SUBQ $4, CX
	JMP  add64_avx2_w4

add64_avx2_w2:
	CMPQ CX, $2
	JL   add64_avx2_tail

	VMOVUPD (SI), X0
	VADDPD  (DX), X0, X0
	VMOVUPD X0, (DI)
	ADDQ $16, SI
	ADDQ $16, DX
	ADDQ $16, DI
	SUBQ $2, CX
	JMP  add64_avx2_w2

add64_avx2_tail:
	TESTQ CX, CX
	JZ   add64_avx2_done

add64_avx2_tail_loop:
	MOVSD (SI), X0
	VADDSD (DX), X0, X0
	MOVSD X0, (DI)
	ADDQ $8, SI
	ADDQ $8, DX
	ADDQ $8, DI
	DECQ CX
	JNZ  add64_avx2_tail_loop

add64_avx2_done:
	VZEROUPPER
	RET

// func AddScalarFloat64AVX2Asm(dst, src *float64, offset float64, count int)
TEXT ·AddScalarFloat64AVX2Asm(SB), NOSPLIT, $0-40
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVSD offset+16(FP), X0
	MOVQ count+24(FP), CX
	TESTQ CX, CX
	JZ   adds64_avx2_done

	VBROADCASTSD X0, Y0

adds64_avx2_w4:
	CMPQ CX, $4
	JL   adds64_avx2_w2

	VMOVUPD (SI), Y1
	VADDPD  Y0, Y1, Y1
	VMOVUPD Y1, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $4, CX
	JMP  adds64_avx2_w4

adds64_avx2_w2:
	CMPQ CX, $2
	JL   adds64_avx2_tail

	VMOVUPD (SI), X1
	VADDPD  X0, X1, X1
	VMOVUPD X1, (DI)
	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $2, CX
	JMP  adds64_avx2_w2

adds64_avx2_tail:
	TESTQ CX, CX
	JZ   adds64_avx2_done

adds64_avx2_tail_loop:
	MOVSD (SI), X1
	VADDSD X0, X1, X1
	MOVSD X1, (DI)
	ADDQ $8, SI
	ADDQ $8, DI
	DECQ CX
	JNZ  adds64_avx2_tail_loop

adds64_avx2_done:
	VZEROUPPER
	RET

// func SqrtFloat64AVX2Asm(dst, src *float64, count int)
TEXT ·SqrtFloat64AVX2Asm(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ count+16(FP), CX
	TESTQ CX, CX
	JZ   sqrt64_avx2_done

sqrt64_avx2_w4:
	CMPQ CX, $4
	JL   sqrt64_avx2_w2

	VMOVUPD (SI), Y0
	VSQRTPD Y0, Y0
	VMOVUPD Y0, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $4, CX
	JMP  sqrt64_avx2_w4

sqrt64_avx2_w2:
	CMPQ CX, $2
	JL   sqrt64_avx2_tail

	VMOVUPD (SI), X0
	VSQRTPD X0, X0
	VMOVUPD X0, (DI)
	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $2, CX
	JMP  sqrt64_avx2_w2

sqrt64_avx2_tail:
	TESTQ CX, CX
	JZ   sqrt64_avx2_done

sqrt64_avx2_tail_loop:
	MOVSD (SI), X0
	VSQRTSD X0, X0, X0
	MOVSD X0, (DI)
	ADDQ $8, SI
	ADDQ $8, DI
	DECQ CX
	JNZ  sqrt64_avx2_tail_loop

sqrt64_avx2_done:
	VZEROUPPER
	RET

// func MaxFloat64AVX2Asm(src *float64, count int) float64
TEXT ·MaxFloat64AVX2Asm(SB), NOSPLIT, $0-24
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX
	TESTQ CX, CX
	JZ   max64_avx2_zero

	MOVSD (SI), X0
	DECQ CX
	JZ   max64_avx2_done

max64_avx2_loop:
	MOVSD (SI), X1
	VMAXSD X1, X0, X0
	ADDQ $8, SI
	DECQ CX
	JNZ  max64_avx2_loop

max64_avx2_done:
	MOVSD X0, ret+16(FP)
	RET

max64_avx2_zero:
	XORPS X0, X0
	MOVSD X0, ret+16(FP)
	RET
