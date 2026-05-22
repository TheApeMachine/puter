#include "textflag.h"

DATA ewZeroSSE2<>+0(SB)/4, $0x00000000
DATA ewAbsMaskSSE2<>+0(SB)/4, $0x7fffffff
DATA ewSignMaskSSE2<>+0(SB)/4, $0x80000000
GLOBL ewZeroSSE2<>(SB), RODATA|NOPTR, $4
GLOBL ewAbsMaskSSE2<>(SB), RODATA|NOPTR, $4
GLOBL ewSignMaskSSE2<>(SB), RODATA|NOPTR, $4

#define WIDEN_BF16_4(src, xLo, xHi) \
	MOVDQU X2, (src); \
	VPXOR  X3, X3; \
	VPUNPCKLWD X3, X2, xLo; \
	VPUNPCKHWD X3, X2, xHi; \
	VPSLLD $16, xLo; \
	VPSLLD $16, xHi

#define NARROW_BF16_4(xLo, xHi, dst) \
	VPSRLD $16, xLo; \
	VPSRLD $16, xHi; \
	MOVL  xLo, AX; \
	MOVW  AX, (dst); \
	PEXTRD $1, xLo, AX; \
	MOVW  AX, 2(dst); \
	MOVL  xHi, AX; \
	MOVW  AX, 4(dst); \
	PEXTRD $1, xHi, AX; \
	MOVW  AX, 6(dst)

#define BF16_BIN_LOOP(opLo, opHi, opScalar) \
	MOVQ dst+0(FP), DI; \
	MOVQ left+8(FP), SI; \
	MOVQ right+16(FP), R8; \
	MOVQ n+24(FP), CX; \
w4: \
	CMPQ CX, $4; \
	JL   tail; \
	WIDEN_BF16_4(SI, X4, X5); \
	WIDEN_BF16_4(R8, X6, X7); \
	opLo; \
	opHi; \
	NARROW_BF16_4(X4, X5, DI); \
	ADDQ $8, SI; \
	ADDQ $8, R8; \
	ADDQ $8, DI; \
	SUBQ $4, CX; \
	JMP  w4; \
tail: \
	TESTQ CX, CX; \
	JZ   done; \
scalar: \
	MOVWLZX (SI), AX; \
	SHLQ  $16, AX; \
	VMOVD X2, AX; \
	MOVWLZX (R8), DX; \
	SHLQ  $16, DX; \
	VMOVD X3, DX; \
	opScalar; \
	VPSRLD $16, X2, X2; \
	MOVL  X2, AX; \
	MOVW  AX, (DI); \
	ADDQ $2, SI; \
	ADDQ $2, R8; \
	ADDQ $2, DI; \
	DECQ CX; \
	JNZ  scalar; \
done: \
	RET

#define MAX_PS(a, b, m, t) \
	CMPPS  $6, b, a, m; \
	ANDPS  m, a; \
	ANDNPS m, b, t; \
	ORPS   t, a

#define MIN_PS(a, b, m, t) \
	CMPPS  $6, a, b, m; \
	ANDPS  m, b; \
	ANDNPS m, a, t; \
	ORPS   t, b; \
	MOVAPS b, a

// func AddBFloat16SSE2Asm(dst, left, right *uint16, n int)
TEXT ·AddBFloat16SSE2Asm(SB), NOSPLIT, $0-32
	BF16_BIN_LOOP(ADDPS X6, X4; , ADDPS X7, X5; , VADDSS X3, X2, X2)

// func SubBFloat16SSE2Asm(dst, left, right *uint16, n int)
TEXT ·SubBFloat16SSE2Asm(SB), NOSPLIT, $0-32
	BF16_BIN_LOOP(SUBPS X6, X4; , SUBPS X7, X5; , VSUBSS X3, X2, X2)

// func MulBFloat16SSE2Asm(dst, left, right *uint16, n int)
TEXT ·MulBFloat16SSE2Asm(SB), NOSPLIT, $0-32
	BF16_BIN_LOOP(MULPS X6, X4; , MULPS X7, X5; , VMULSS X3, X2, X2)

// func DivBFloat16SSE2Asm(dst, left, right *uint16, n int)
TEXT ·DivBFloat16SSE2Asm(SB), NOSPLIT, $0-32
	BF16_BIN_LOOP(DIVPS X6, X4; , DIVPS X7, X5; , VDIVSS X3, X2, X2)

// func MaxBFloat16SSE2Asm(dst, left, right *uint16, n int)
TEXT ·MaxBFloat16SSE2Asm(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ left+8(FP), SI
	MOVQ right+16(FP), R8
	MOVQ n+24(FP), CX
max_w4:
	CMPQ CX, $4
	JL   max_tail
	WIDEN_BF16_4(SI, X4, X5)
	WIDEN_BF16_4(R8, X6, X7)
	MAX_PS(X4, X6, X8, X9)
	MAX_PS(X5, X7, X8, X9)
	NARROW_BF16_4(X4, X5, DI)
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
	SHLQ  $16, AX
	VMOVD X2, AX
	MOVWLZX (R8), DX
	SHLQ  $16, DX
	VMOVD X3, DX
	MAX_PS(X2, X3, X4, X5)
	VPSRLD $16, X2, X2
	MOVL  X2, AX
	MOVW  AX, (DI)
	ADDQ $2, SI
	ADDQ $2, R8
	ADDQ $2, DI
	DECQ CX
	JNZ  max_scalar
max_done:
	RET

// func MinBFloat16SSE2Asm(dst, left, right *uint16, n int)
TEXT ·MinBFloat16SSE2Asm(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ left+8(FP), SI
	MOVQ right+16(FP), R8
	MOVQ n+24(FP), CX
min_w4:
	CMPQ CX, $4
	JL   min_tail
	WIDEN_BF16_4(SI, X4, X5)
	WIDEN_BF16_4(R8, X6, X7)
	MIN_PS(X4, X6, X8, X9)
	MIN_PS(X5, X7, X8, X9)
	NARROW_BF16_4(X4, X5, DI)
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
	SHLQ  $16, AX
	VMOVD X2, AX
	MOVWLZX (R8), DX
	SHLQ  $16, DX
	VMOVD X3, DX
	MIN_PS(X2, X3, X4, X5)
	VPSRLD $16, X2, X2
	MOVL  X2, AX
	MOVW  AX, (DI)
	ADDQ $2, SI
	ADDQ $2, R8
	ADDQ $2, DI
	DECQ CX
	JNZ  min_scalar
min_done:
	RET

// func AbsBFloat16SSE2Asm(dst, src *uint16, n int)
TEXT ·AbsBFloat16SSE2Asm(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX
	MOVSS ewAbsMaskSSE2<>(SB), X10
abs_w4:
	CMPQ CX, $4
	JL   abs_tail
	WIDEN_BF16_4(SI, X4, X5)
	ANDPS X10, X4
	ANDPS X10, X5
	NARROW_BF16_4(X4, X5, DI)
	ADDQ $8, SI
	ADDQ $8, DI
	SUBQ $4, CX
	JMP  abs_w4
abs_tail:
	TESTQ CX, CX
	JZ   abs_done
abs_scalar:
	MOVWLZX (SI), AX
	SHLQ  $16, AX
	VMOVD X2, AX
	ANDPS X10, X2
	VPSRLD $16, X2, X2
	MOVL  X2, AX
	MOVW  AX, (DI)
	ADDQ $2, SI
	ADDQ $2, DI
	DECQ CX
	JNZ  abs_scalar
abs_done:
	RET

// func NegBFloat16SSE2Asm(dst, src *uint16, n int)
TEXT ·NegBFloat16SSE2Asm(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX
	MOVSS ewSignMaskSSE2<>(SB), X10
neg_w4:
	CMPQ CX, $4
	JL   neg_tail
	WIDEN_BF16_4(SI, X4, X5)
	XORPS X10, X4
	XORPS X10, X5
	NARROW_BF16_4(X4, X5, DI)
	ADDQ $8, SI
	ADDQ $8, DI
	SUBQ $4, CX
	JMP  neg_w4
neg_tail:
	TESTQ CX, CX
	JZ   neg_done
neg_scalar:
	MOVWLZX (SI), AX
	SHLQ  $16, AX
	VMOVD X2, AX
	XORPS X10, X2
	VPSRLD $16, X2, X2
	MOVL  X2, AX
	MOVW  AX, (DI)
	ADDQ $2, SI
	ADDQ $2, DI
	DECQ CX
	JNZ  neg_scalar
neg_done:
	RET

// func SqrtBFloat16SSE2Asm(dst, src *uint16, n int)
TEXT ·SqrtBFloat16SSE2Asm(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX
sqrt_w4:
	CMPQ CX, $4
	JL   sqrt_tail
	WIDEN_BF16_4(SI, X4, X5)
	SQRTPS X4, X4
	SQRTPS X5, X5
	NARROW_BF16_4(X4, X5, DI)
	ADDQ $8, SI
	ADDQ $8, DI
	SUBQ $4, CX
	JMP  sqrt_w4
sqrt_tail:
	TESTQ CX, CX
	JZ   sqrt_done
sqrt_scalar:
	MOVWLZX (SI), AX
	SHLQ  $16, AX
	VMOVD X2, AX
	SQRTSS X2, X2, X2
	VPSRLD $16, X2, X2
	MOVL  X2, AX
	MOVW  AX, (DI)
	ADDQ $2, SI
	ADDQ $2, DI
	DECQ CX
	JNZ  sqrt_scalar
sqrt_done:
	RET

// func ReluBFloat16SSE2Asm(dst, src *uint16, n int)
TEXT ·ReluBFloat16SSE2Asm(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX
	XORPS X10, X10
relu_w4:
	CMPQ CX, $4
	JL   relu_tail
	WIDEN_BF16_4(SI, X4, X5)
	CMPPS $6, X10, X4, X8
	ANDPS X8, X4
	CMPPS $6, X10, X5, X8
	ANDPS X8, X5
	NARROW_BF16_4(X4, X5, DI)
	ADDQ $8, SI
	ADDQ $8, DI
	SUBQ $4, CX
	JMP  relu_w4
relu_tail:
	TESTQ CX, CX
	JZ   relu_done
relu_scalar:
	MOVWLZX (SI), AX
	SHLQ  $16, AX
	VMOVD X2, AX
	CMPPS $6, X10, X2, X4
	ANDPS X4, X2
	VPSRLD $16, X2, X2
	MOVL  X2, AX
	MOVW  AX, (DI)
	ADDQ $2, SI
	ADDQ $2, DI
	DECQ CX
	JNZ  relu_scalar
relu_done:
	RET

// func AxpyBFloat16SSE2Asm(y, x *uint16, alpha float32, n int)
TEXT ·AxpyBFloat16SSE2Asm(SB), NOSPLIT, $0-32
	MOVQ y+0(FP), DI
	MOVQ x+8(FP), SI
	MOVSS alpha+16(FP), X15
	MOVQ n+24(FP), CX
	SHUFPS $0, X15, X15
axpy_w4:
	CMPQ CX, $4
	JL   axpy_tail
	WIDEN_BF16_4(DI, X4, X5)
	WIDEN_BF16_4(SI, X6, X7)
	MULPS X15, X6
	MULPS X15, X7
	ADDPS X6, X4
	ADDPS X7, X5
	NARROW_BF16_4(X4, X5, DI)
	ADDQ $8, DI
	ADDQ $8, SI
	SUBQ $4, CX
	JMP  axpy_w4
axpy_tail:
	TESTQ CX, CX
	JZ   axpy_done
axpy_scalar:
	MOVWLZX (DI), AX
	SHLQ  $16, AX
	VMOVD X2, AX
	MOVWLZX (SI), DX
	SHLQ  $16, DX
	VMOVD X3, DX
	MULSS X15, X3
	VADDSS X3, X2, X2
	VPSRLD $16, X2, X2
	MOVL  X2, AX
	MOVW  AX, (DI)
	ADDQ $2, DI
	ADDQ $2, SI
	DECQ CX
	JNZ  axpy_scalar
axpy_done:
	RET
