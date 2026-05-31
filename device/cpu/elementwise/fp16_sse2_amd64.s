#include "textflag.h"
#include "../f16c_fp16_macros.inc"

DATA ewAbsMaskFP16SSE2<>+0(SB)/4, $0x7fffffff
DATA ewSignMaskFP16SSE2<>+0(SB)/4, $0x80000000
GLOBL ewAbsMaskFP16SSE2<>(SB), RODATA|NOPTR, $4
GLOBL ewSignMaskFP16SSE2<>(SB), RODATA|NOPTR, $4

#define MAX_PS(a, b, m, t) \
	VCMPPS  $6, b, a, m; \
	ANDPS  m, a; \
	VANDNPS m, b, t; \
	ORPS   t, a

#define MIN_PS(a, b, m, t) \
	VCMPPS  $6, a, b, m; \
	ANDPS  m, b; \
	VANDNPS m, a, t; \
	ORPS   t, b; \
	MOVAPS b, a

// func AddFloat16SSE2Asm(dst, left, right *uint16, n int)
TEXT ·AddFloat16SSE2Asm(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ left+8(FP), SI
	MOVQ right+16(FP), R8
	MOVQ n+24(FP), CX
add_w4:
	CMPQ CX, $4
	JL   add_tail
	FP16_WIDEN_SSE2_4(SI, X4)
	FP16_WIDEN_SSE2_4(R8, X6)
	ADDPS X6, X4
	FP16_NARROW_SSE2_4(X4, DI)
	ADDQ $8, SI
	ADDQ $8, R8
	ADDQ $8, DI
	SUBQ $4, CX
	JMP  add_w4
add_tail:
	TESTQ CX, CX
	JZ   add_done
add_scalar:
	MOVWLZX (SI), AX
	VMOVD X2, AX
	VCVTPH2PS X2, X2
	MOVWLZX (R8), DX
	VMOVD X3, DX
	VCVTPH2PS X3, X3
	VADDSS X3, X2, X2
	VMOVAPS X2, X0
	VCVTPS2PH_X0_X2
	MOVL X2, AX
	MOVW AX, (DI)
	ADDQ $2, SI
	ADDQ $2, R8
	ADDQ $2, DI
	DECQ CX
	JNZ  add_scalar
add_done:
	RET

// func SubFloat16SSE2Asm(dst, left, right *uint16, n int)
TEXT ·SubFloat16SSE2Asm(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ left+8(FP), SI
	MOVQ right+16(FP), R8
	MOVQ n+24(FP), CX
sub_w4:
	CMPQ CX, $4
	JL   sub_tail
	FP16_WIDEN_SSE2_4(SI, X4)
	FP16_WIDEN_SSE2_4(R8, X6)
	SUBPS X6, X4
	FP16_NARROW_SSE2_4(X4, DI)
	ADDQ $8, SI
	ADDQ $8, R8
	ADDQ $8, DI
	SUBQ $4, CX
	JMP  sub_w4
sub_tail:
	TESTQ CX, CX
	JZ   sub_done
sub_scalar:
	MOVWLZX (SI), AX
	VMOVD X2, AX
	VCVTPH2PS X2, X2
	MOVWLZX (R8), DX
	VMOVD X3, DX
	VCVTPH2PS X3, X3
	VSUBSS X3, X2, X2
	VMOVAPS X2, X0
	VCVTPS2PH_X0_X2
	MOVL X2, AX
	MOVW AX, (DI)
	ADDQ $2, SI
	ADDQ $2, R8
	ADDQ $2, DI
	DECQ CX
	JNZ  sub_scalar
sub_done:
	RET

// func MulFloat16SSE2Asm(dst, left, right *uint16, n int)
TEXT ·MulFloat16SSE2Asm(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ left+8(FP), SI
	MOVQ right+16(FP), R8
	MOVQ n+24(FP), CX
mul_w4:
	CMPQ CX, $4
	JL   mul_tail
	FP16_WIDEN_SSE2_4(SI, X4)
	FP16_WIDEN_SSE2_4(R8, X6)
	MULPS X6, X4
	FP16_NARROW_SSE2_4(X4, DI)
	ADDQ $8, SI
	ADDQ $8, R8
	ADDQ $8, DI
	SUBQ $4, CX
	JMP  mul_w4
mul_tail:
	TESTQ CX, CX
	JZ   mul_done
mul_scalar:
	MOVWLZX (SI), AX
	VMOVD X2, AX
	VCVTPH2PS X2, X2
	MOVWLZX (R8), DX
	VMOVD X3, DX
	VCVTPH2PS X3, X3
	VMULSS X3, X2, X2
	VMOVAPS X2, X0
	VCVTPS2PH_X0_X2
	MOVL X2, AX
	MOVW AX, (DI)
	ADDQ $2, SI
	ADDQ $2, R8
	ADDQ $2, DI
	DECQ CX
	JNZ  mul_scalar
mul_done:
	RET

// func DivFloat16SSE2Asm(dst, left, right *uint16, n int)
TEXT ·DivFloat16SSE2Asm(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ left+8(FP), SI
	MOVQ right+16(FP), R8
	MOVQ n+24(FP), CX
div_w4:
	CMPQ CX, $4
	JL   div_tail
	FP16_WIDEN_SSE2_4(SI, X4)
	FP16_WIDEN_SSE2_4(R8, X6)
	DIVPS X6, X4
	FP16_NARROW_SSE2_4(X4, DI)
	ADDQ $8, SI
	ADDQ $8, R8
	ADDQ $8, DI
	SUBQ $4, CX
	JMP  div_w4
div_tail:
	TESTQ CX, CX
	JZ   div_done
div_scalar:
	MOVWLZX (SI), AX
	VMOVD X2, AX
	VCVTPH2PS X2, X2
	MOVWLZX (R8), DX
	VMOVD X3, DX
	VCVTPH2PS X3, X3
	VDIVSS X3, X2, X2
	VMOVAPS X2, X0
	VCVTPS2PH_X0_X2
	MOVL X2, AX
	MOVW AX, (DI)
	ADDQ $2, SI
	ADDQ $2, R8
	ADDQ $2, DI
	DECQ CX
	JNZ  div_scalar
div_done:
	RET

// func MaxFloat16SSE2Asm(dst, left, right *uint16, n int)
TEXT ·MaxFloat16SSE2Asm(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ left+8(FP), SI
	MOVQ right+16(FP), R8
	MOVQ n+24(FP), CX
max_w4:
	CMPQ CX, $4
	JL   max_tail
	FP16_WIDEN_SSE2_4(SI, X4)
	FP16_WIDEN_SSE2_4(R8, X6)
	VCMPPS  $6, X6, X4, X8
	ANDPS  X8, X4
	VANDNPS X9, X8, X6
	ORPS   X9, X4
	FP16_NARROW_SSE2_4(X4, DI)
	ADDQ $8, SI
	ADDQ $8, R8
	ADDQ $8, DI
	SUBQ $4, CX
	JMP  max_w4
max_tail:
	TESTQ CX, CX
	JZ   max_done
max_scalar:
	MOVWLZX (SI), AX
	VMOVD X2, AX
	VCVTPH2PS X2, X2
	MOVWLZX (R8), DX
	VMOVD X3, DX
	VCVTPH2PS X3, X3
	VCMPPS  $6, X3, X2, X4
	ANDPS  X4, X2
	VANDNPS X5, X4, X3
	ORPS   X5, X2
	VMOVAPS X2, X0
	VCVTPS2PH_X0_X2
	MOVL X2, AX
	MOVW AX, (DI)
	ADDQ $2, SI
	ADDQ $2, R8
	ADDQ $2, DI
	DECQ CX
	JNZ  max_scalar
max_done:
	RET

// func MinFloat16SSE2Asm(dst, left, right *uint16, n int)
TEXT ·MinFloat16SSE2Asm(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ left+8(FP), SI
	MOVQ right+16(FP), R8
	MOVQ n+24(FP), CX
min_w4:
	CMPQ CX, $4
	JL   min_tail
	FP16_WIDEN_SSE2_4(SI, X4)
	FP16_WIDEN_SSE2_4(R8, X6)
	VCMPPS  $6, X4, X6, X8
	ANDPS  X8, X6
	VANDNPS X9, X8, X4
	ORPS   X9, X6
	MOVAPS X6, X4
	FP16_NARROW_SSE2_4(X4, DI)
	ADDQ $8, SI
	ADDQ $8, R8
	ADDQ $8, DI
	SUBQ $4, CX
	JMP  min_w4
min_tail:
	TESTQ CX, CX
	JZ   min_done
min_scalar:
	MOVWLZX (SI), AX
	VMOVD X2, AX
	VCVTPH2PS X2, X2
	MOVWLZX (R8), DX
	VMOVD X3, DX
	VCVTPH2PS X3, X3
	VCMPPS  $6, X2, X3, X4
	ANDPS  X4, X3
	VANDNPS X5, X4, X2
	ORPS   X5, X3
	MOVAPS X3, X2
	VMOVAPS X2, X0
	VCVTPS2PH_X0_X2
	MOVL X2, AX
	MOVW AX, (DI)
	ADDQ $2, SI
	ADDQ $2, R8
	ADDQ $2, DI
	DECQ CX
	JNZ  min_scalar
min_done:
	RET

// func AbsFloat16SSE2Asm(dst, src *uint16, n int)
TEXT ·AbsFloat16SSE2Asm(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX
	MOVSS ewAbsMaskFP16SSE2<>(SB), X10
abs_w4:
	CMPQ CX, $4
	JL   abs_tail
	FP16_WIDEN_SSE2_4(SI, X4)
	ANDPS X10, X4
	FP16_NARROW_SSE2_4(X4, DI)
	ADDQ $8, SI
	ADDQ $8, DI
	SUBQ $4, CX
	JMP  abs_w4
abs_tail:
	TESTQ CX, CX
	JZ   abs_done
abs_scalar:
	MOVWLZX (SI), AX
	VMOVD X2, AX
	VCVTPH2PS X2, X2
	ANDPS X10, X2
	VMOVAPS X2, X0
	VCVTPS2PH_X0_X2
	MOVL X2, AX
	MOVW AX, (DI)
	ADDQ $2, SI
	ADDQ $2, DI
	DECQ CX
	JNZ  abs_scalar
abs_done:
	RET

// func NegFloat16SSE2Asm(dst, src *uint16, n int)
TEXT ·NegFloat16SSE2Asm(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX
	MOVSS ewSignMaskFP16SSE2<>(SB), X10
neg_w4:
	CMPQ CX, $4
	JL   neg_tail
	FP16_WIDEN_SSE2_4(SI, X4)
	VXORPS X10, X4, X4
	FP16_NARROW_SSE2_4(X4, DI)
	ADDQ $8, SI
	ADDQ $8, DI
	SUBQ $4, CX
	JMP  neg_w4
neg_tail:
	TESTQ CX, CX
	JZ   neg_done
neg_scalar:
	MOVWLZX (SI), AX
	VMOVD X2, AX
	VCVTPH2PS X2, X2
	VXORPS X10, X2, X2
	VMOVAPS X2, X0
	VCVTPS2PH_X0_X2
	MOVL X2, AX
	MOVW AX, (DI)
	ADDQ $2, SI
	ADDQ $2, DI
	DECQ CX
	JNZ  neg_scalar
neg_done:
	RET

// func SqrtFloat16SSE2Asm(dst, src *uint16, n int)
TEXT ·SqrtFloat16SSE2Asm(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX
sqrt_w4:
	CMPQ CX, $4
	JL   sqrt_tail
	FP16_WIDEN_SSE2_4(SI, X4)
	SQRTPS X4, X4
	FP16_NARROW_SSE2_4(X4, DI)
	ADDQ $8, SI
	ADDQ $8, DI
	SUBQ $4, CX
	JMP  sqrt_w4
sqrt_tail:
	TESTQ CX, CX
	JZ   sqrt_done
sqrt_scalar:
	MOVWLZX (SI), AX
	VMOVD X2, AX
	VCVTPH2PS X2, X2
	SQRTSS X2, X2
	VMOVAPS X2, X0
	VCVTPS2PH_X0_X2
	MOVL X2, AX
	MOVW AX, (DI)
	ADDQ $2, SI
	ADDQ $2, DI
	DECQ CX
	JNZ  sqrt_scalar
sqrt_done:
	RET

// func ReluFloat16SSE2Asm(dst, src *uint16, n int)
TEXT ·ReluFloat16SSE2Asm(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX
	VXORPS X10, X10, X10
relu_w4:
	CMPQ CX, $4
	JL   relu_tail
	FP16_WIDEN_SSE2_4(SI, X4)
	VCMPPS $6, X10, X4, X8
	ANDPS X8, X4
	FP16_NARROW_SSE2_4(X4, DI)
	ADDQ $8, SI
	ADDQ $8, DI
	SUBQ $4, CX
	JMP  relu_w4
relu_tail:
	TESTQ CX, CX
	JZ   relu_done
relu_scalar:
	MOVWLZX (SI), AX
	VMOVD X2, AX
	VCVTPH2PS X2, X2
	VCMPPS $6, X10, X2, X4
	ANDPS X4, X2
	VMOVAPS X2, X0
	VCVTPS2PH_X0_X2
	MOVL X2, AX
	MOVW AX, (DI)
	ADDQ $2, SI
	ADDQ $2, DI
	DECQ CX
	JNZ  relu_scalar
relu_done:
	RET

// func AxpyFloat16SSE2Asm(y, x *uint16, alpha float32, n int)
TEXT ·AxpyFloat16SSE2Asm(SB), NOSPLIT, $0-32
	MOVQ y+0(FP), DI
	MOVQ x+8(FP), SI
	MOVSS alpha+16(FP), X15
	MOVQ n+24(FP), CX
	SHUFPS $0, X15, X15
axpy_w4:
	CMPQ CX, $4
	JL   axpy_tail
	FP16_WIDEN_SSE2_4(DI, X4)
	FP16_WIDEN_SSE2_4(SI, X6)
	MULPS X15, X6
	ADDPS X6, X4
	FP16_NARROW_SSE2_4(X4, DI)
	ADDQ $8, DI
	ADDQ $8, SI
	SUBQ $4, CX
	JMP  axpy_w4
axpy_tail:
	TESTQ CX, CX
	JZ   axpy_done
axpy_scalar:
	MOVWLZX (DI), AX
	VMOVD X2, AX
	VCVTPH2PS X2, X2
	MOVWLZX (SI), DX
	VMOVD X3, DX
	VCVTPH2PS X3, X3
	MULSS X15, X3
	VADDSS X3, X2, X2
	VMOVAPS X2, X0
	VCVTPS2PH_X0_X2
	MOVL X2, AX
	MOVW AX, (DI)
	ADDQ $2, DI
	ADDQ $2, SI
	DECQ CX
	JNZ  axpy_scalar
axpy_done:
	RET
