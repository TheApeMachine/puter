#include "textflag.h"

DATA ewAbsMaskFP16AVX512<>+0(SB)/2, $0x7fff
DATA ewSignMaskFP16AVX512<>+0(SB)/2, $0x8000
GLOBL ewAbsMaskFP16AVX512<>(SB), RODATA|NOPTR, $2
GLOBL ewSignMaskFP16AVX512<>(SB), RODATA|NOPTR, $2

#define VCVTPS2PH_X0_X2 WORD $0xC4E3; WORD $0x7D1D; BYTE $0xD0; BYTE $0x00

#define FP16_BINARY_AVX512(op_y16, op_x8, op_scalar) \
	MOVQ dst+0(FP), DI \
	MOVQ left+8(FP), SI \
	MOVQ right+16(FP), R8 \
	MOVQ n+24(FP), CX \
loop16: \
	CMPQ CX, $16 \
	JL   loop8 \
	VMOVUPH Y0, (SI) \
	VMOVUPH Y1, (R8) \
	op_y16 \
	VMOVUPH Y0, (DI) \
	ADDQ $32, SI \
	ADDQ $32, R8 \
	ADDQ $32, DI \
	SUBQ $16, CX \
	JMP  loop16 \
loop8: \
	CMPQ CX, $8 \
	JL   scalar_tail \
	VMOVUPH X0, (SI) \
	VMOVUPH X1, (R8) \
	op_x8 \
	VMOVUPH X0, (DI) \
	ADDQ $16, SI \
	ADDQ $16, R8 \
	ADDQ $16, DI \
	SUBQ $8, CX \
	JMP  loop8 \
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

#define ADD_Y16 VADDPH Y1, Y0, Y0
#define ADD_X8  VADDPH X1, X0, X0
#define ADD_SCALAR VADDPH X3, X2, X2

#define SUB_Y16 VSUBPH Y1, Y0, Y0
#define SUB_X8  VSUBPH X1, X0, X0
#define SUB_SCALAR VSUBPH X3, X2, X2

#define MUL_Y16 VMULPH Y1, Y0, Y0
#define MUL_X8  VMULPH X1, X0, X0
#define MUL_SCALAR VMULPH X3, X2, X2

#define DIV_Y16 VDIVPH Y1, Y0, Y0
#define DIV_X8  VDIVPH X1, X0, X0
#define DIV_SCALAR VDIVPH X3, X2, X2

#define MAX_Y16 \
	VMAXPH Y1, Y0, Y0

#define MAX_X8 \
	VMAXPH X1, X0, X0

#define MAX_SCALAR \
	VMAXPH X3, X2, X2

#define MIN_Y16 \
	VMINPH Y1, Y0, Y0

#define MIN_X8 \
	VMINPH X1, X0, X0

#define MIN_SCALAR \
	VMINPH X3, X2, X2

// func AddFloat16AVX512Asm(dst, left, right *uint16, n int)
TEXT ·AddFloat16AVX512Asm(SB), NOSPLIT, $0-32
	FP16_BINARY_AVX512(ADD_Y16, ADD_X8, ADD_SCALAR)

// func SubFloat16AVX512Asm(dst, left, right *uint16, n int)
TEXT ·SubFloat16AVX512Asm(SB), NOSPLIT, $0-32
	FP16_BINARY_AVX512(SUB_Y16, SUB_X8, SUB_SCALAR)

// func MulFloat16AVX512Asm(dst, left, right *uint16, n int)
TEXT ·MulFloat16AVX512Asm(SB), NOSPLIT, $0-32
	FP16_BINARY_AVX512(MUL_Y16, MUL_X8, MUL_SCALAR)

// func DivFloat16AVX512Asm(dst, left, right *uint16, n int)
TEXT ·DivFloat16AVX512Asm(SB), NOSPLIT, $0-32
	FP16_BINARY_AVX512(DIV_Y16, DIV_X8, DIV_SCALAR)

// func MaxFloat16AVX512Asm(dst, left, right *uint16, n int)
TEXT ·MaxFloat16AVX512Asm(SB), NOSPLIT, $0-32
	FP16_BINARY_AVX512(MAX_Y16, MAX_X8, MAX_SCALAR)

// func MinFloat16AVX512Asm(dst, left, right *uint16, n int)
TEXT ·MinFloat16AVX512Asm(SB), NOSPLIT, $0-32
	FP16_BINARY_AVX512(MIN_Y16, MIN_X8, MIN_SCALAR)

// func AbsFloat16AVX512Asm(dst, src *uint16, n int)
TEXT ·AbsFloat16AVX512Asm(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX
	VPBROADCASTW ewAbsMaskFP16AVX512<>(SB), Y30
abs_w16:
	CMPQ CX, $16
	JL   abs_w8
	VMOVUPH Y0, (SI)
	VPAND Y30, Y0, Y0
	VMOVUPH Y0, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $16, CX
	JMP  abs_w16
abs_w8:
	CMPQ CX, $8
	JL   abs_tail
	VMOVUPH X0, (SI)
	VPAND X30, X0, X0
	VMOVUPH X0, (DI)
	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $8, CX
	JMP  abs_w8
abs_tail:
	TESTQ CX, CX
	JZ   abs_done
	VPBROADCASTW ewAbsMaskFP16AVX512<>(SB), X30
abs_scalar:
	VPBROADCASTW (SI), X2
	VPAND X30, X2, X2
	VMOVD X2, AX
	MOVW AX, (DI)
	ADDQ $2, SI
	ADDQ $2, DI
	DECQ CX
	JNZ  abs_scalar
abs_done:
	RET

// func NegFloat16AVX512Asm(dst, src *uint16, n int)
TEXT ·NegFloat16AVX512Asm(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX
	VPBROADCASTW ewSignMaskFP16AVX512<>(SB), Y30
neg_w16:
	CMPQ CX, $16
	JL   neg_w8
	VMOVUPH Y0, (SI)
	VPXOR Y30, Y0, Y0
	VMOVUPH Y0, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $16, CX
	JMP  neg_w16
neg_w8:
	CMPQ CX, $8
	JL   neg_tail
	VMOVUPH X0, (SI)
	VPXOR X30, X0, X0
	VMOVUPH X0, (DI)
	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $8, CX
	JMP  neg_w8
neg_tail:
	TESTQ CX, CX
	JZ   neg_done
	VPBROADCASTW ewSignMaskFP16AVX512<>(SB), X30
neg_scalar:
	VPBROADCASTW (SI), X2
	VPXOR X30, X2, X2
	VMOVD X2, AX
	MOVW AX, (DI)
	ADDQ $2, SI
	ADDQ $2, DI
	DECQ CX
	JNZ  neg_scalar
neg_done:
	RET

// func SqrtFloat16AVX512Asm(dst, src *uint16, n int)
TEXT ·SqrtFloat16AVX512Asm(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX
sqrt_w16:
	CMPQ CX, $16
	JL   sqrt_w8
	VMOVUPH Y0, (SI)
	VSQRTPH Y0, Y0
	VMOVUPH Y0, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $16, CX
	JMP  sqrt_w16
sqrt_w8:
	CMPQ CX, $8
	JL   sqrt_tail
	VMOVUPH X0, (SI)
	VSQRTPH X0, X0
	VMOVUPH X0, (DI)
	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $8, CX
	JMP  sqrt_w8
sqrt_tail:
	TESTQ CX, CX
	JZ   sqrt_done
sqrt_scalar:
	VPBROADCASTW (SI), X2
	VSQRTPH X2, X2
	VMOVD X2, AX
	MOVW AX, (DI)
	ADDQ $2, SI
	ADDQ $2, DI
	DECQ CX
	JNZ  sqrt_scalar
sqrt_done:
	RET

// func ReluFloat16AVX512Asm(dst, src *uint16, n int)
TEXT ·ReluFloat16AVX512Asm(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX
	VPXORD Y31, Y31, Y31
relu_w16:
	CMPQ CX, $16
	JL   relu_w8
	VMOVUPH Y0, (SI)
	VCMPPH K1, Y0, Y31, $13
	VBLENDMPH Y0, Y31, Y0, K1
	VMOVUPH Y0, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $16, CX
	JMP  relu_w16
relu_w8:
	CMPQ CX, $8
	JL   relu_tail
	VMOVUPH X0, (SI)
	VPXORD X31, X31, X31
	VCMPPH K1, X0, X31, $13
	VBLENDMPH X0, X31, X0, K1
	VMOVUPH X0, (DI)
	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $8, CX
	JMP  relu_w8
relu_tail:
	TESTQ CX, CX
	JZ   relu_done
	VPXORD X31, X31, X31
relu_scalar:
	VPBROADCASTW (SI), X2
	VCMPPH K1, X2, X31, $13
	VBLENDMPH X2, X31, X2, K1
	VMOVD X2, AX
	MOVW AX, (DI)
	ADDQ $2, SI
	ADDQ $2, DI
	DECQ CX
	JNZ  relu_scalar
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
	VPBROADCASTW X2, Y31
	VEXTRACTI128 $0, Y31, X31
axpy_w16:
	CMPQ CX, $16
	JL   axpy_w8
	VMOVUPH Y0, (DI)
	VMOVUPH Y1, (SI)
	VFMADD231PH Y31, Y1, Y0
	VMOVUPH Y0, (DI)
	ADDQ $32, DI
	ADDQ $32, SI
	SUBQ $16, CX
	JMP  axpy_w16
axpy_w8:
	CMPQ CX, $8
	JL   axpy_tail
	VMOVUPH X0, (DI)
	VMOVUPH X1, (SI)
	VFMADD231PH X31, X1, X0
	VMOVUPH X0, (DI)
	ADDQ $16, DI
	ADDQ $16, SI
	SUBQ $8, CX
	JMP  axpy_w8
axpy_tail:
	TESTQ CX, CX
	JZ   axpy_done
axpy_scalar:
	VPBROADCASTW (DI), X0
	VPBROADCASTW (SI), X1
	VFMADD231PH X31, X1, X0
	VMOVD X0, AX
	MOVW AX, (DI)
	ADDQ $2, DI
	ADDQ $2, SI
	DECQ CX
	JNZ  axpy_scalar
axpy_done:
	RET
