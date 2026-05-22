#include "textflag.h"

DATA ewZeroAVX512<>+0(SB)/4, $0x00000000
GLOBL ewZeroAVX512<>(SB), RODATA|NOPTR, $4

DATA ewAbsMaskAVX512<>+0(SB)/4, $0x7fffffff
GLOBL ewAbsMaskAVX512<>(SB), RODATA|NOPTR, $4

DATA ewSignMaskAVX512<>+0(SB)/4, $0x80000000
GLOBL ewSignMaskAVX512<>(SB), RODATA|NOPTR, $4

#define WIDEN_BF16_8H(baseReg, dstY) \
	VMOVDQU X2, (baseReg); \
	VPMOVZXWD X2, dstY; \
	VPSLLD $16, dstY, dstY; \
	VPSRLDQ $8, X2, X3; \
	VPMOVZXWD X3, Y4; \
	VPSLLD $16, Y4, Y4; \
	VEXTRACTI128 $0, Y4, X4; \
	VINSERTF128 $1, X4, dstY, dstY

#define NARROW_BF16_Y8(dstReg) \
	VPSRLD $16, Y0, Y0; \
	VEXTRACTI128 $0, Y0, X2; \
	MOVL  X2, AX; \
	MOVW  AX, (dstReg); \
	PEXTRD $1, X2, AX; \
	MOVW  AX, 2(dstReg); \
	PEXTRD $2, X2, AX; \
	MOVW  AX, 4(dstReg); \
	PEXTRD $3, X2, AX; \
	MOVW  AX, 6(dstReg); \
	VEXTRACTI128 $1, Y0, X2; \
	MOVL  X2, AX; \
	MOVW  AX, 8(dstReg); \
	PEXTRD $1, X2, AX; \
	MOVW  AX, 10(dstReg); \
	PEXTRD $2, X2, AX; \
	MOVW  AX, 12(dstReg); \
	PEXTRD $3, X2, AX; \
	MOVW  AX, 14(dstReg)

#define WIDEN_BF16_4H(baseReg, dstY) \
	VMOVDQU X2, (baseReg); \
	VPMOVZXWD X2, dstY; \
	VPSLLD $16, dstY, dstY

#define NARROW_BF16_Y4(dstReg) \
	VPSRLD $16, Y0, Y0; \
	VEXTRACTI128 $0, Y0, X2; \
	MOVL  X2, AX; \
	MOVW  AX, (dstReg); \
	PEXTRD $1, X2, AX; \
	MOVW  AX, 2(dstReg); \
	PEXTRD $2, X2, AX; \
	MOVW  AX, 4(dstReg); \
	PEXTRD $3, X2, AX; \
	MOVW  AX, 6(dstReg)

// func AddBFloat16AVX512Asm(dst, left, right *uint16, n int)
TEXT ·AddBFloat16AVX512Asm(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ left+8(FP), SI
	MOVQ right+16(FP), R8
	MOVQ n+24(FP), CX
add_w8:
	CMPQ CX, $8
	JL   add_w4
	WIDEN_BF16_8H(SI, Y0)
	WIDEN_BF16_8H(R8, Y1)
	VADDPS Y1, Y0, Y0
	NARROW_BF16_Y8(DI)
	ADDQ $16, SI
	ADDQ $16, R8
	ADDQ $16, DI
	SUBQ $8, CX
	JMP  add_w8
add_w4:
	CMPQ CX, $4
	JL   add_tail
	WIDEN_BF16_4H(SI, Y0)
	WIDEN_BF16_4H(R8, Y1)
	VADDPS Y1, Y0, Y0
	NARROW_BF16_Y4(DI)
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
	SHLQ  $16, AX
	VMOVD X2, AX
	MOVWLZX (R8), DX
	SHLQ  $16, DX
	VMOVD X3, DX
	VADDSS X3, X2, X2
	VPSRLD $16, X2, X2
	MOVL  X2, AX
	MOVW  AX, (DI)
	ADDQ $2, SI
	ADDQ $2, R8
	ADDQ $2, DI
	DECQ CX
	JNZ  add_scalar
add_done:
	RET

// func SubBFloat16AVX512Asm(dst, left, right *uint16, n int)
TEXT ·SubBFloat16AVX512Asm(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ left+8(FP), SI
	MOVQ right+16(FP), R8
	MOVQ n+24(FP), CX
sub_w8:
	CMPQ CX, $8
	JL   sub_w4
	WIDEN_BF16_8H(SI, Y0)
	WIDEN_BF16_8H(R8, Y1)
	VSUBPS Y1, Y0, Y0
	NARROW_BF16_Y8(DI)
	ADDQ $16, SI
	ADDQ $16, R8
	ADDQ $16, DI
	SUBQ $8, CX
	JMP  sub_w8
sub_w4:
	CMPQ CX, $4
	JL   sub_tail
	WIDEN_BF16_4H(SI, Y0)
	WIDEN_BF16_4H(R8, Y1)
	VSUBPS Y1, Y0, Y0
	NARROW_BF16_Y4(DI)
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
	SHLQ  $16, AX
	VMOVD X2, AX
	MOVWLZX (R8), DX
	SHLQ  $16, DX
	VMOVD X3, DX
	VSUBSS X3, X2, X2
	VPSRLD $16, X2, X2
	MOVL  X2, AX
	MOVW  AX, (DI)
	ADDQ $2, SI
	ADDQ $2, R8
	ADDQ $2, DI
	DECQ CX
	JNZ  sub_scalar
sub_done:
	RET

// func MulBFloat16AVX512Asm(dst, left, right *uint16, n int)
TEXT ·MulBFloat16AVX512Asm(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ left+8(FP), SI
	MOVQ right+16(FP), R8
	MOVQ n+24(FP), CX
mul_w8:
	CMPQ CX, $8
	JL   mul_w4
	WIDEN_BF16_8H(SI, Y0)
	WIDEN_BF16_8H(R8, Y1)
	VMULPS Y1, Y0, Y0
	NARROW_BF16_Y8(DI)
	ADDQ $16, SI
	ADDQ $16, R8
	ADDQ $16, DI
	SUBQ $8, CX
	JMP  mul_w8
mul_w4:
	CMPQ CX, $4
	JL   mul_tail
	WIDEN_BF16_4H(SI, Y0)
	WIDEN_BF16_4H(R8, Y1)
	VMULPS Y1, Y0, Y0
	NARROW_BF16_Y4(DI)
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
	SHLQ  $16, AX
	VMOVD X2, AX
	MOVWLZX (R8), DX
	SHLQ  $16, DX
	VMOVD X3, DX
	VMULSS X3, X2, X2
	VPSRLD $16, X2, X2
	MOVL  X2, AX
	MOVW  AX, (DI)
	ADDQ $2, SI
	ADDQ $2, R8
	ADDQ $2, DI
	DECQ CX
	JNZ  mul_scalar
mul_done:
	RET

// func DivBFloat16AVX512Asm(dst, left, right *uint16, n int)
TEXT ·DivBFloat16AVX512Asm(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ left+8(FP), SI
	MOVQ right+16(FP), R8
	MOVQ n+24(FP), CX
div_w8:
	CMPQ CX, $8
	JL   div_w4
	WIDEN_BF16_8H(SI, Y0)
	WIDEN_BF16_8H(R8, Y1)
	VDIVPS Y1, Y0, Y0
	NARROW_BF16_Y8(DI)
	ADDQ $16, SI
	ADDQ $16, R8
	ADDQ $16, DI
	SUBQ $8, CX
	JMP  div_w8
div_w4:
	CMPQ CX, $4
	JL   div_tail
	WIDEN_BF16_4H(SI, Y0)
	WIDEN_BF16_4H(R8, Y1)
	VDIVPS Y1, Y0, Y0
	NARROW_BF16_Y4(DI)
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
	SHLQ  $16, AX
	VMOVD X2, AX
	MOVWLZX (R8), DX
	SHLQ  $16, DX
	VMOVD X3, DX
	VDIVSS X3, X2, X2
	VPSRLD $16, X2, X2
	MOVL  X2, AX
	MOVW  AX, (DI)
	ADDQ $2, SI
	ADDQ $2, R8
	ADDQ $2, DI
	DECQ CX
	JNZ  div_scalar
div_done:
	RET

// func MaxBFloat16AVX512Asm(dst, left, right *uint16, n int)
TEXT ·MaxBFloat16AVX512Asm(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ left+8(FP), SI
	MOVQ right+16(FP), R8
	MOVQ n+24(FP), CX
max_w8:
	CMPQ CX, $8
	JL   max_w4
	WIDEN_BF16_8H(SI, Y0)
	WIDEN_BF16_8H(R8, Y1)
	VCMPPS $6, Y0, Y1, Y2
	VBLENDVPS Y1, Y0, Y2, Y0
	NARROW_BF16_Y8(DI)
	ADDQ $16, SI
	ADDQ $16, R8
	ADDQ $16, DI
	SUBQ $8, CX
	JMP  max_w8
max_w4:
	CMPQ CX, $4
	JL   max_tail
	WIDEN_BF16_4H(SI, Y0)
	WIDEN_BF16_4H(R8, Y1)
	VCMPPS $6, Y0, Y1, Y2
	VBLENDVPS Y1, Y0, Y2, Y0
	NARROW_BF16_Y4(DI)
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
	VCMPSS $6, X2, X3, X4
	VBLENDVPS X3, X2, X4, X2
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

// func MinBFloat16AVX512Asm(dst, left, right *uint16, n int)
TEXT ·MinBFloat16AVX512Asm(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ left+8(FP), SI
	MOVQ right+16(FP), R8
	MOVQ n+24(FP), CX
min_w8:
	CMPQ CX, $8
	JL   min_w4
	WIDEN_BF16_8H(SI, Y0)
	WIDEN_BF16_8H(R8, Y1)
	VCMPPS $6, Y1, Y0, Y2
	VBLENDVPS Y1, Y0, Y2, Y0
	NARROW_BF16_Y8(DI)
	ADDQ $16, SI
	ADDQ $16, R8
	ADDQ $16, DI
	SUBQ $8, CX
	JMP  min_w8
min_w4:
	CMPQ CX, $4
	JL   min_tail
	WIDEN_BF16_4H(SI, Y0)
	WIDEN_BF16_4H(R8, Y1)
	VCMPPS $6, Y1, Y0, Y2
	VBLENDVPS Y1, Y0, Y2, Y0
	NARROW_BF16_Y4(DI)
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
	VCMPSS $6, X3, X2, X4
	VBLENDVPS X3, X2, X4, X2
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

// func AbsBFloat16AVX512Asm(dst, src *uint16, n int)
TEXT ·AbsBFloat16AVX512Asm(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX
	VBROADCASTSS ewAbsMaskAVX512<>(SB), Y1
abs_w8:
	CMPQ CX, $8
	JL   abs_w4
	WIDEN_BF16_8H(SI, Y0)
	VANDPS Y1, Y0, Y0
	NARROW_BF16_Y8(DI)
	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $8, CX
	JMP  abs_w8
abs_w4:
	CMPQ CX, $4
	JL   abs_tail
	WIDEN_BF16_4H(SI, Y0)
	VANDPS X1, X0, X0
	NARROW_BF16_Y4(DI)
	ADDQ $8, SI
	ADDQ $8, DI
	SUBQ $4, CX
	JMP  abs_w4
abs_tail:
	TESTQ CX, CX
	JZ   abs_done
	VBROADCASTSS ewAbsMaskAVX512<>(SB), X1
abs_scalar:
	MOVWLZX (SI), AX
	SHLQ  $16, AX
	VMOVD X2, AX
	VANDPS X1, X2, X2
	VPSRLD $16, X2, X2
	MOVL  X2, AX
	MOVW  AX, (DI)
	ADDQ $2, SI
	ADDQ $2, DI
	DECQ CX
	JNZ  abs_scalar
abs_done:
	RET

// func NegBFloat16AVX512Asm(dst, src *uint16, n int)
TEXT ·NegBFloat16AVX512Asm(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX
	VBROADCASTSS ewSignMaskAVX512<>(SB), Y1
neg_w8:
	CMPQ CX, $8
	JL   neg_w4
	WIDEN_BF16_8H(SI, Y0)
	VXORPS Y1, Y0, Y0
	NARROW_BF16_Y8(DI)
	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $8, CX
	JMP  neg_w8
neg_w4:
	CMPQ CX, $4
	JL   neg_tail
	WIDEN_BF16_4H(SI, Y0)
	VXORPS X1, X0, X0
	NARROW_BF16_Y4(DI)
	ADDQ $8, SI
	ADDQ $8, DI
	SUBQ $4, CX
	JMP  neg_w4
neg_tail:
	TESTQ CX, CX
	JZ   neg_done
	VBROADCASTSS ewSignMaskAVX512<>(SB), X1
neg_scalar:
	MOVWLZX (SI), AX
	SHLQ  $16, AX
	VMOVD X2, AX
	VXORPS X1, X2, X2
	VPSRLD $16, X2, X2
	MOVL  X2, AX
	MOVW  AX, (DI)
	ADDQ $2, SI
	ADDQ $2, DI
	DECQ CX
	JNZ  neg_scalar
neg_done:
	RET

// func SqrtBFloat16AVX512Asm(dst, src *uint16, n int)
TEXT ·SqrtBFloat16AVX512Asm(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX
sqrt_w8:
	CMPQ CX, $8
	JL   sqrt_w4
	WIDEN_BF16_8H(SI, Y0)
	VSQRTPS Y0, Y0
	NARROW_BF16_Y8(DI)
	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $8, CX
	JMP  sqrt_w8
sqrt_w4:
	CMPQ CX, $4
	JL   sqrt_tail
	WIDEN_BF16_4H(SI, Y0)
	VSQRTPS X0, X0
	NARROW_BF16_Y4(DI)
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
	VSQRTSS X2, X2, X2
	VPSRLD $16, X2, X2
	MOVL  X2, AX
	MOVW  AX, (DI)
	ADDQ $2, SI
	ADDQ $2, DI
	DECQ CX
	JNZ  sqrt_scalar
sqrt_done:
	RET

// func ReluBFloat16AVX512Asm(dst, src *uint16, n int)
TEXT ·ReluBFloat16AVX512Asm(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX
	VBROADCASTSS ewZeroAVX512<>(SB), Y1
relu_w8:
	CMPQ CX, $8
	JL   relu_w4
	WIDEN_BF16_8H(SI, Y0)
	VCMPPS $6, Y0, Y1, Y2
	VBLENDVPS Y1, Y0, Y2, Y0
	NARROW_BF16_Y8(DI)
	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $8, CX
	JMP  relu_w8
relu_w4:
	CMPQ CX, $4
	JL   relu_tail
	WIDEN_BF16_4H(SI, Y0)
	VCMPPS $6, Y0, Y1, Y2
	VBLENDVPS Y1, Y0, Y2, Y0
	NARROW_BF16_Y4(DI)
	ADDQ $8, SI
	ADDQ $8, DI
	SUBQ $4, CX
	JMP  relu_w4
relu_tail:
	TESTQ CX, CX
	JZ   relu_done
	VXORPS X1, X1, X1
relu_scalar:
	MOVWLZX (SI), AX
	SHLQ  $16, AX
	VMOVD X2, AX
	VCMPSS $6, X2, X1, X4
	VBLENDVPS X1, X2, X4, X2
	VPSRLD $16, X2, X2
	MOVL  X2, AX
	MOVW  AX, (DI)
	ADDQ $2, SI
	ADDQ $2, DI
	DECQ CX
	JNZ  relu_scalar
relu_done:
	RET

// func AxpyBFloat16AVX512Asm(y, x *uint16, alpha float32, n int)
TEXT ·AxpyBFloat16AVX512Asm(SB), NOSPLIT, $0-32
	MOVQ y+0(FP), DI
	MOVQ x+8(FP), SI
	MOVSS alpha+16(FP), X15
	MOVQ n+24(FP), CX
axpybf16_w8:
	CMPQ CX, $8
	JL   axpybf16_w4
	WIDEN_BF16_8H(DI, Y0)
	WIDEN_BF16_8H(SI, Y1)
	VBROADCASTSS X15, Y31
	VFMADD231PS Y31, Y1, Y0
	NARROW_BF16_Y8(DI)
	ADDQ $16, DI
	ADDQ $16, SI
	SUBQ $8, CX
	JMP  axpybf16_w8
axpybf16_w4:
	CMPQ CX, $4
	JL   axpybf16_tail
	WIDEN_BF16_4H(DI, Y0)
	WIDEN_BF16_4H(SI, Y1)
	VBROADCASTSS X15, Y31
	VFMADD231PS Y31, Y1, Y0
	NARROW_BF16_Y4(DI)
	ADDQ $8, DI
	ADDQ $8, SI
	SUBQ $4, CX
	JMP  axpybf16_w4
axpybf16_tail:
	TESTQ CX, CX
	JZ   axpybf16_done
axpybf16_scalar:
	MOVWLZX (DI), AX
	SHLQ  $16, AX
	VMOVD X2, AX
	MOVWLZX (SI), DX
	SHLQ  $16, DX
	VMOVD X3, DX
	VMULSS X15, X3, X3
	VADDSS X3, X2, X2
	VPSRLD $16, X2, X2
	MOVL  X2, AX
	MOVW  AX, (DI)
	ADDQ $2, DI
	ADDQ $2, SI
	DECQ CX
	JNZ  axpybf16_scalar
axpybf16_done:
	RET
