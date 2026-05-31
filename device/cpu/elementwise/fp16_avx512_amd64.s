#include "textflag.h"
#include "../avx512_fp16_macros.inc"
#include "../f16c_fp16_macros.inc"

DATA ewAbsMaskHalfAVX512<>+0(SB)/2, $0x7fff
DATA ewSignMaskHalfAVX512<>+0(SB)/2, $0x8000
GLOBL ewAbsMaskHalfAVX512<>(SB), RODATA|NOPTR, $2
GLOBL ewSignMaskHalfAVX512<>(SB), RODATA|NOPTR, $2

#define FP16_BINARY_AVX512(op_y8, op_x4, op_scalar) \
	MOVQ dst+0(FP), DI \
	MOVQ left+8(FP), SI \
	MOVQ right+16(FP), R8 \
	MOVQ n+24(FP), CX \
loop8: \
	CMPQ CX, $8 \
	JL   loop4 \
	VMOVUPH_Y0_SI \
	VMOVUPH_Y1_R8 \
	op_y8 \
	STORE_Y0_16H_DI \
	ADDQ $16, SI \
	ADDQ $16, R8 \
	ADDQ $16, DI \
	SUBQ $8, CX \
	JMP  loop8 \
loop4: \
	CMPQ CX, $4 \
	JL   scalar_tail \
	VMOVUPH_X0_SI \
	VMOVUPH_X1_R8 \
	op_x4 \
	STORE_X0_8H_DI \
	ADDQ $8, SI \
	ADDQ $8, R8 \
	ADDQ $8, DI \
	SUBQ $4, CX \
	JMP  loop4 \
scalar_tail: \
	TESTQ CX, CX \
	JZ   done \
scalar_loop: \
	VPBROADCASTW (SI), X2 \
	VPBROADCASTW (R8), X3 \
	op_scalar \
	VMOVD X2, AX \
	MOVW AX, (DI) \
	ADDQ $2, SI \
	ADDQ $2, R8 \
	ADDQ $2, DI \
	DECQ CX \
	JNZ  scalar_loop \
done: \
	RET

#define ADD_Y8  VADDPH_Y0_Y1_Y0
#define ADD_X4  VADDPH_X0_Y1_X0
#define ADD_SCALAR VADDPH_X2_X3_X2

#define SUB_Y8  VSUBPH_Y0_Y1_Y0
#define SUB_X4  VSUBPH_X0_Y1_X0
#define SUB_SCALAR VSUBPH_X2_X3_X2

#define MUL_Y8  VMULPH_Y0_Y1_Y0
#define MUL_X4  VMULPH_X0_Y1_X0
#define MUL_SCALAR VMULPH_X2_X3_X2

#define DIV_Y8  VDIVPH_Y0_Y1_Y0
#define DIV_X4  VDIVPH_X0_Y1_X0
#define DIV_SCALAR VDIVPH_X2_X3_X2

#define MAX_Y8  VMAXPH_Y0_Y1_Y0
#define MAX_X4  VMAXPH_X0_Y1_X0
#define MAX_SCALAR VMAXPH_X2_X3_X2

#define MIN_Y8  VMINPH_Y0_Y1_Y0
#define MIN_X4  VMINPH_X0_Y1_X0
#define MIN_SCALAR VMINPH_X2_X3_X2

// func AddFloat16AVX512Asm(dst, left, right *uint16, n int)
TEXT ·AddFloat16AVX512Asm(SB), NOSPLIT, $0-32
	FP16_BINARY_AVX512(ADD_Y8, ADD_X4, ADD_SCALAR)

// func SubFloat16AVX512Asm(dst, left, right *uint16, n int)
TEXT ·SubFloat16AVX512Asm(SB), NOSPLIT, $0-32
	FP16_BINARY_AVX512(SUB_Y8, SUB_X4, SUB_SCALAR)

// func MulFloat16AVX512Asm(dst, left, right *uint16, n int)
TEXT ·MulFloat16AVX512Asm(SB), NOSPLIT, $0-32
	FP16_BINARY_AVX512(MUL_Y8, MUL_X4, MUL_SCALAR)

// func DivFloat16AVX512Asm(dst, left, right *uint16, n int)
TEXT ·DivFloat16AVX512Asm(SB), NOSPLIT, $0-32
	FP16_BINARY_AVX512(DIV_Y8, DIV_X4, DIV_SCALAR)

// func MaxFloat16AVX512Asm(dst, left, right *uint16, n int)
TEXT ·MaxFloat16AVX512Asm(SB), NOSPLIT, $0-32
	FP16_BINARY_AVX512(MAX_Y8, MAX_X4, MAX_SCALAR)

// func MinFloat16AVX512Asm(dst, left, right *uint16, n int)
TEXT ·MinFloat16AVX512Asm(SB), NOSPLIT, $0-32
	FP16_BINARY_AVX512(MIN_Y8, MIN_X4, MIN_SCALAR)

#define FP16_UNARY_AVX512(op_y8, op_x4, op_scalar) \
	MOVQ dst+0(FP), DI \
	MOVQ src+8(FP), SI \
	MOVQ n+16(FP), CX \
u_loop8: \
	CMPQ CX, $8 \
	JL   u_loop4 \
	VMOVUPH_Y0_SI \
	op_y8 \
	STORE_Y0_16H_DI \
	ADDQ $16, SI \
	ADDQ $16, DI \
	SUBQ $8, CX \
	JMP  u_loop8 \
u_loop4: \
	CMPQ CX, $4 \
	JL   u_scalar_tail \
	VMOVUPH_X0_SI \
	op_x4 \
	STORE_X0_8H_DI \
	ADDQ $8, SI \
	ADDQ $8, DI \
	SUBQ $4, CX \
	JMP  u_loop4 \
u_scalar_tail: \
	TESTQ CX, CX \
	JZ   u_done \
u_scalar_loop: \
	VPBROADCASTW (SI), X2 \
	op_scalar \
	VMOVD X2, AX \
	MOVW AX, (DI) \
	ADDQ $2, SI \
	ADDQ $2, DI \
	DECQ CX \
	JNZ  u_scalar_loop \
u_done: \
	RET

// func AbsFloat16AVX512Asm(dst, src *uint16, n int)
TEXT ·AbsFloat16AVX512Asm(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX
	VPBROADCASTW ewAbsMaskHalfAVX512<>(SB), Y14
	VPBROADCASTW ewAbsMaskHalfAVX512<>(SB), X14
abs_w8:
	CMPQ CX, $8
	JL   abs_w4
	VMOVUPH_Y0_SI
	VPAND Y14, Y0, Y0
	STORE_Y0_16H_DI
	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $8, CX
	JMP  abs_w8
abs_w4:
	CMPQ CX, $4
	JL   abs_scalar_tail
	VMOVUPH_X0_SI
	VPAND X14, X0, X0
	STORE_X0_8H_DI
	ADDQ $8, SI
	ADDQ $8, DI
	SUBQ $4, CX
	JMP  abs_w4
abs_scalar_tail:
	TESTQ CX, CX
	JZ   abs_done
abs_scalar_loop:
	VPBROADCASTW (SI), X2
	VPAND X14, X2, X2
	VMOVD X2, AX
	MOVW AX, (DI)
	ADDQ $2, SI
	ADDQ $2, DI
	DECQ CX
	JNZ  abs_scalar_loop
abs_done:
	RET

// func NegFloat16AVX512Asm(dst, src *uint16, n int)
TEXT ·NegFloat16AVX512Asm(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX
	VPBROADCASTW ewSignMaskHalfAVX512<>(SB), Y14
	VPBROADCASTW ewSignMaskHalfAVX512<>(SB), X14
neg_w8:
	CMPQ CX, $8
	JL   neg_w4
	VMOVUPH_Y0_SI
	VPXOR Y14, Y0, Y0
	STORE_Y0_16H_DI
	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $8, CX
	JMP  neg_w8
neg_w4:
	CMPQ CX, $4
	JL   neg_scalar_tail
	VMOVUPH_X0_SI
	VPXOR X14, X0, X0
	STORE_X0_8H_DI
	ADDQ $8, SI
	ADDQ $8, DI
	SUBQ $4, CX
	JMP  neg_w4
neg_scalar_tail:
	TESTQ CX, CX
	JZ   neg_done
neg_scalar_loop:
	VPBROADCASTW (SI), X2
	VPXOR X14, X2, X2
	VMOVD X2, AX
	MOVW AX, (DI)
	ADDQ $2, SI
	ADDQ $2, DI
	DECQ CX
	JNZ  neg_scalar_loop
neg_done:
	RET

// func SqrtFloat16AVX512Asm(dst, src *uint16, n int)
TEXT ·SqrtFloat16AVX512Asm(SB), NOSPLIT, $0-24
	FP16_UNARY_AVX512(VSQRTPH_Y0_Y0, VSQRTPH_X0_X0, VSQRTPH_X0_X0)

// func ReluFloat16AVX512Asm(dst, src *uint16, n int)
TEXT ·ReluFloat16AVX512Asm(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX
	VPXORD Y31, Y31, Y31
	VPXORD X31, X31, X31
relu_w8:
	CMPQ CX, $8
	JL   relu_w4
	VMOVUPH_Y0_SI
	VMAXPH_Y0_Y31_Y0
	STORE_Y0_16H_DI
	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $8, CX
	JMP  relu_w8
relu_w4:
	CMPQ CX, $4
	JL   relu_scalar_tail
	VMOVUPH_X0_SI
	VMAXPH_X0_X31_X0
	STORE_X0_8H_DI
	ADDQ $8, SI
	ADDQ $8, DI
	SUBQ $4, CX
	JMP  relu_w4
relu_scalar_tail:
	TESTQ CX, CX
	JZ   relu_done
relu_scalar_loop:
	VPBROADCASTW (SI), X2
	VMAXPH_X2_X31_X2
	VMOVD X2, AX
	MOVW AX, (DI)
	ADDQ $2, SI
	ADDQ $2, DI
	DECQ CX
	JNZ  relu_scalar_loop
relu_done:
	RET

// func AxpyFloat16AVX512Asm(y, x *uint16, alpha float32, n int)
TEXT ·AxpyFloat16AVX512Asm(SB), NOSPLIT, $0-32
	MOVQ y+0(FP), DI
	MOVQ x+8(FP), SI
	MOVSS alpha+16(FP), X15
	MOVQ n+24(FP), CX

	VMOVAPS X15, X0
	VCVTPS2PH_X0_X2
	VPBROADCASTW_Y14_X14

axpy_w8:
	CMPQ CX, $8
	JL   axpy_w4

	VMOVUPH_Y0_DI
	VMOVUPH_Y1_SI
	VMULPH_Y3_Y1_Y14
	VADDPH_Y0_Y3_Y0
	STORE_Y0_16H_DI

	ADDQ $16, DI
	ADDQ $16, SI
	SUBQ $8, CX
	JMP  axpy_w8

axpy_w4:
	CMPQ CX, $4
	JL   axpy_scalar_tail

	VMOVUPH_X0_DI
	VMOVUPH_X1_SI
	VMULPH_X3_X1_X14
	VADDPH_X0_X3_X0
	STORE_X0_8H_DI

	ADDQ $8, DI
	ADDQ $8, SI
	SUBQ $4, CX
	JMP  axpy_w4

axpy_scalar_tail:
	TESTQ CX, CX
	JZ   axpy_done

axpy_scalar_loop:
	VPBROADCASTW (DI), X0
	VPBROADCASTW (SI), X1
	VMULPH_X3_X1_X14
	VADDPH_X0_X3_X0
	VMOVD X0, AX
	MOVW AX, (DI)

	ADDQ $2, DI
	ADDQ $2, SI
	DECQ CX
	JNZ  axpy_scalar_loop

axpy_done:
	RET
