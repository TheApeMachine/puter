#include "textflag.h"

// func SumFloat64AVX512Asm(src *float64, count int) float64
TEXT ·SumFloat64AVX512Asm(SB), NOSPLIT, $0-24
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX
	TESTQ CX, CX
	JZ   sum64_avx512_zero

	VXORPD Z0, Z0, Z0

sum64_avx512_w8:
	CMPQ CX, $8
	JL   sum64_avx512_w4

	VMOVUPD (SI), Z1
	VADDPD  Z1, Z0, Z0
	ADDQ $64, SI
	SUBQ $8, CX
	JMP  sum64_avx512_w8

sum64_avx512_w4:
	CMPQ CX, $4
	JL   sum64_avx512_reduce

	VMOVUPD (SI), Y1
	VADDPD  Y1, Y0, Y0
	ADDQ $32, SI
	SUBQ $4, CX
	JMP  sum64_avx512_w4

sum64_avx512_reduce:
	VEXTRACTF64X4 $1, Z0, Y1
	VADDPD        Y1, Y0, Y0
	VEXTRACTF128  $1, Y0, X1
	VADDPD        X1, X0, X0
	VHADDPD       X0, X0, X0

	TESTQ CX, CX
	JZ   sum64_avx512_done

sum64_avx512_tail:
	MOVSD (SI), X1
	VADDSD X1, X0, X0
	ADDQ $8, SI
	DECQ CX
	JNZ  sum64_avx512_tail

sum64_avx512_done:
	VZEROUPPER
	MOVSD X0, ret+16(FP)
	RET

sum64_avx512_zero:
	MOVSD $0.0, ret+16(FP)
	RET

// func SumOfSquaresFloat64AVX512Asm(src *float64, count int) float64
TEXT ·SumOfSquaresFloat64AVX512Asm(SB), NOSPLIT, $0-24
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX
	TESTQ CX, CX
	JZ   sumsq64_avx512_zero

	VXORPD Z0, Z0, Z0

sumsq64_avx512_w8:
	CMPQ CX, $8
	JL   sumsq64_avx512_w4

	VMOVUPD (SI), Z1
	VMULPD  Z1, Z1, Z1
	VADDPD  Z1, Z0, Z0
	ADDQ $64, SI
	SUBQ $8, CX
	JMP  sumsq64_avx512_w8

sumsq64_avx512_w4:
	CMPQ CX, $4
	JL   sumsq64_avx512_reduce

	VMOVUPD (SI), Y1
	VMULPD  Y1, Y1, Y1
	VADDPD  Y1, Y0, Y0
	ADDQ $32, SI
	SUBQ $4, CX
	JMP  sumsq64_avx512_w4

sumsq64_avx512_reduce:
	VEXTRACTF64X4 $1, Z0, Y1
	VADDPD        Y1, Y0, Y0
	VEXTRACTF128  $1, Y0, X1
	VADDPD        X1, X0, X0
	VHADDPD       X0, X0, X0

	TESTQ CX, CX
	JZ   sumsq64_avx512_done

sumsq64_avx512_tail:
	MOVSD (SI), X1
	VMULSD X1, X1, X1
	VADDSD X1, X0, X0
	ADDQ $8, SI
	DECQ CX
	JNZ  sumsq64_avx512_tail

sumsq64_avx512_done:
	VZEROUPPER
	MOVSD X0, ret+16(FP)
	RET

sumsq64_avx512_zero:
	MOVSD $0.0, ret+16(FP)
	RET

// func DotFloat64AVX512Asm(left, right *float64, count int) float64
TEXT ·DotFloat64AVX512Asm(SB), NOSPLIT, $0-32
	MOVQ left+0(FP), SI
	MOVQ right+8(FP), DI
	MOVQ count+16(FP), CX
	TESTQ CX, CX
	JZ   dot64_avx512_zero

	VXORPD Z0, Z0, Z0

dot64_avx512_w8:
	CMPQ CX, $8
	JL   dot64_avx512_w4

	VMOVUPD (SI), Z1
	VMOVUPD (DI), Z2
	VMULPD  Z2, Z1, Z1
	VADDPD  Z1, Z0, Z0
	ADDQ $64, SI
	ADDQ $64, DI
	SUBQ $8, CX
	JMP  dot64_avx512_w8

dot64_avx512_w4:
	CMPQ CX, $4
	JL   dot64_avx512_reduce

	VMOVUPD (SI), Y1
	VMOVUPD (DI), Y2
	VMULPD  Y2, Y1, Y1
	VADDPD  Y1, Y0, Y0
	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $4, CX
	JMP  dot64_avx512_w4

dot64_avx512_reduce:
	VEXTRACTF64X4 $1, Z0, Y1
	VADDPD        Y1, Y0, Y0
	VEXTRACTF128  $1, Y0, X1
	VADDPD        X1, X0, X0
	VHADDPD       X0, X0, X0

	TESTQ CX, CX
	JZ   dot64_avx512_done

dot64_avx512_tail:
	MOVSD (SI), X1
	VMULSD (DI), X1, X1
	VADDSD X1, X0, X0
	ADDQ $8, SI
	ADDQ $8, DI
	DECQ CX
	JNZ  dot64_avx512_tail

dot64_avx512_done:
	VZEROUPPER
	MOVSD X0, ret+24(FP)
	RET

dot64_avx512_zero:
	MOVSD $0.0, ret+24(FP)
	RET

// func ScaleFloat64AVX512Asm(dst, src *float64, scale float64, count int)
TEXT ·ScaleFloat64AVX512Asm(SB), NOSPLIT, $0-40
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVSD scale+16(FP), X0
	MOVQ count+24(FP), CX
	TESTQ CX, CX
	JZ   scale64_avx512_done

	VBROADCASTSD X0, Z0

scale64_avx512_w8:
	CMPQ CX, $8
	JL   scale64_avx512_w4

	VMOVUPD (SI), Z1
	VMULPD  Z0, Z1, Z1
	VMOVUPD Z1, (DI)
	ADDQ $64, SI
	ADDQ $64, DI
	SUBQ $8, CX
	JMP  scale64_avx512_w8

scale64_avx512_w4:
	CMPQ CX, $4
	JL   scale64_avx512_tail

	VBROADCASTSD X0, Y0
	VMOVUPD (SI), Y1
	VMULPD  Y0, Y1, Y1
	VMOVUPD Y1, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $4, CX
	JMP  scale64_avx512_w4

scale64_avx512_tail:
	TESTQ CX, CX
	JZ   scale64_avx512_done

scale64_avx512_tail_loop:
	MOVSD (SI), X1
	VMULSD X0, X1, X1
	MOVSD X1, (DI)
	ADDQ $8, SI
	ADDQ $8, DI
	DECQ CX
	JNZ  scale64_avx512_tail_loop

scale64_avx512_done:
	VZEROUPPER
	RET

// func MulFloat64AVX512Asm(dst, left, right *float64, count int)
TEXT ·MulFloat64AVX512Asm(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ left+8(FP), SI
	MOVQ right+16(FP), DX
	MOVQ count+24(FP), CX
	TESTQ CX, CX
	JZ   mul64_avx512_done

mul64_avx512_w8:
	CMPQ CX, $8
	JL   mul64_avx512_w4

	VMOVUPD (SI), Z0
	VMOVUPD (DX), Z1
	VMULPD  Z1, Z0, Z0
	VMOVUPD Z0, (DI)
	ADDQ $64, SI
	ADDQ $64, DX
	ADDQ $64, DI
	SUBQ $8, CX
	JMP  mul64_avx512_w8

mul64_avx512_w4:
	CMPQ CX, $4
	JL   mul64_avx512_tail

	VMOVUPD (SI), Y0
	VMOVUPD (DX), Y1
	VMULPD  Y1, Y0, Y0
	VMOVUPD Y0, (DI)
	ADDQ $32, SI
	ADDQ $32, DX
	ADDQ $32, DI
	SUBQ $4, CX
	JMP  mul64_avx512_w4

mul64_avx512_tail:
	TESTQ CX, CX
	JZ   mul64_avx512_done

mul64_avx512_tail_loop:
	MOVSD (SI), X0
	VMULSD (DX), X0, X0
	MOVSD X0, (DI)
	ADDQ $8, SI
	ADDQ $8, DX
	ADDQ $8, DI
	DECQ CX
	JNZ  mul64_avx512_tail_loop

mul64_avx512_done:
	VZEROUPPER
	RET

// func AddFloat64AVX512Asm(dst, left, right *float64, count int)
TEXT ·AddFloat64AVX512Asm(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ left+8(FP), SI
	MOVQ right+16(FP), DX
	MOVQ count+24(FP), CX
	TESTQ CX, CX
	JZ   add64_avx512_done

add64_avx512_w8:
	CMPQ CX, $8
	JL   add64_avx512_w4

	VMOVUPD (SI), Z0
	VADDPD  (DX), Z0, Z0
	VMOVUPD Z0, (DI)
	ADDQ $64, SI
	ADDQ $64, DX
	ADDQ $64, DI
	SUBQ $8, CX
	JMP  add64_avx512_w8

add64_avx512_w4:
	CMPQ CX, $4
	JL   add64_avx512_tail

	VMOVUPD (SI), Y0
	VADDPD  (DX), Y0, Y0
	VMOVUPD Y0, (DI)
	ADDQ $32, SI
	ADDQ $32, DX
	ADDQ $32, DI
	SUBQ $4, CX
	JMP  add64_avx512_w4

add64_avx512_tail:
	TESTQ CX, CX
	JZ   add64_avx512_done

add64_avx512_tail_loop:
	MOVSD (SI), X0
	VADDSD (DX), X0, X0
	MOVSD X0, (DI)
	ADDQ $8, SI
	ADDQ $8, DX
	ADDQ $8, DI
	DECQ CX
	JNZ  add64_avx512_tail_loop

add64_avx512_done:
	VZEROUPPER
	RET

// func AddScalarFloat64AVX512Asm(dst, src *float64, offset float64, count int)
TEXT ·AddScalarFloat64AVX512Asm(SB), NOSPLIT, $0-40
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVSD offset+16(FP), X0
	MOVQ count+24(FP), CX
	TESTQ CX, CX
	JZ   adds64_avx512_done

	VBROADCASTSD X0, Z0

adds64_avx512_w8:
	CMPQ CX, $8
	JL   adds64_avx512_w4

	VMOVUPD (SI), Z1
	VADDPD  Z0, Z1, Z1
	VMOVUPD Z1, (DI)
	ADDQ $64, SI
	ADDQ $64, DI
	SUBQ $8, CX
	JMP  adds64_avx512_w8

adds64_avx512_w4:
	CMPQ CX, $4
	JL   adds64_avx512_tail

	VBROADCASTSD X0, Y0
	VMOVUPD (SI), Y1
	VADDPD  Y0, Y1, Y1
	VMOVUPD Y1, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $4, CX
	JMP  adds64_avx512_w4

adds64_avx512_tail:
	TESTQ CX, CX
	JZ   adds64_avx512_done

adds64_avx512_tail_loop:
	MOVSD (SI), X1
	VADDSD X0, X1, X1
	MOVSD X1, (DI)
	ADDQ $8, SI
	ADDQ $8, DI
	DECQ CX
	JNZ  adds64_avx512_tail_loop

adds64_avx512_done:
	VZEROUPPER
	RET

// func SqrtFloat64AVX512Asm(dst, src *float64, count int)
TEXT ·SqrtFloat64AVX512Asm(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ count+16(FP), CX
	TESTQ CX, CX
	JZ   sqrt64_avx512_done

sqrt64_avx512_w8:
	CMPQ CX, $8
	JL   sqrt64_avx512_w4

	VMOVUPD (SI), Z0
	VSQRTPD Z0, Z0
	VMOVUPD Z0, (DI)
	ADDQ $64, SI
	ADDQ $64, DI
	SUBQ $8, CX
	JMP  sqrt64_avx512_w8

sqrt64_avx512_w4:
	CMPQ CX, $4
	JL   sqrt64_avx512_tail

	VMOVUPD (SI), Y0
	VSQRTPD Y0, Y0
	VMOVUPD Y0, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $4, CX
	JMP  sqrt64_avx512_w4

sqrt64_avx512_tail:
	TESTQ CX, CX
	JZ   sqrt64_avx512_done

sqrt64_avx512_tail_loop:
	MOVSD (SI), X0
	VSQRTSD X0, X0, X0
	MOVSD X0, (DI)
	ADDQ $8, SI
	ADDQ $8, DI
	DECQ CX
	JNZ  sqrt64_avx512_tail_loop

sqrt64_avx512_done:
	VZEROUPPER
	RET

// func MaxFloat64AVX512Asm(src *float64, count int) float64
TEXT ·MaxFloat64AVX512Asm(SB), NOSPLIT, $0-24
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX
	TESTQ CX, CX
	JZ   max64_avx512_zero

	MOVSD (SI), X0
	DECQ CX
	JZ   max64_avx512_done

max64_avx512_loop:
	MOVSD (SI), X1
	VMAXSD X1, X0, X0
	ADDQ $8, SI
	DECQ CX
	JNZ  max64_avx512_loop

max64_avx512_done:
	MOVSD X0, ret+16(FP)
	RET

max64_avx512_zero:
	MOVSD $0.0, ret+16(FP)
	RET
