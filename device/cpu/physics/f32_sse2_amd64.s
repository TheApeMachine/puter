#include "textflag.h"

DATA phySse2C16<>+0(SB)/4, $0x41800000
GLOBL phySse2C16<>(SB), RODATA|NOPTR, $4

DATA phySse2Cm30<>+0(SB)/4, $0xC1F00000
GLOBL phySse2Cm30<>(SB), RODATA|NOPTR, $4

DATA phySse2Cm1<>+0(SB)/4, $0xBF800000
GLOBL phySse2Cm1<>(SB), RODATA|NOPTR, $4

// func Laplacian1DStencilF32SSE2Asm(out, left, center, right *float32, invH2 float32, n int)
TEXT ·Laplacian1DStencilF32SSE2Asm(SB), NOSPLIT, $0-48
	MOVQ out+0(FP), R8
	MOVQ left+8(FP), R9
	MOVQ center+16(FP), R10
	MOVQ right+24(FP), R11
	MOVSS invH2+32(FP), X7

	MOVQ n+40(FP), CX
	TESTQ CX, CX
	JZ   lap1_sse2_done

lap1_sse2_w4:
	CMPQ CX, $4
	JL   lap1_sse2_tail

	VMOVUPS (R9), X0
	VMOVUPS (R11), X1
	VADDPS  X1, X0, X0
	VMOVUPS (R10), X1
	VADDPS  X1, X1, X1
	VSUBPS  X1, X0, X0
	SHUFPS  $0, X7, X7
	MULPS   X7, X0
	VMOVUPS X0, (R8)

	ADDQ $16, R8
	ADDQ $16, R9
	ADDQ $16, R10
	ADDQ $16, R11
	SUBQ $4, CX
	JMP  lap1_sse2_w4

lap1_sse2_tail:
	TESTQ CX, CX
	JZ   lap1_sse2_done

lap1_sse2_scalar:
	MOVSS (R9), X0
	MOVSS (R11), X1
	ADDSS X1, X0
	MOVSS (R10), X1
	ADDSS X1, X1
	SUBSS X1, X0
	MULSS X7, X0
	MOVSS X0, (R8)
	ADDQ  $4, R8
	ADDQ  $4, R9
	ADDQ  $4, R10
	ADDQ  $4, R11
	DECQ  CX
	JNZ  lap1_sse2_scalar

lap1_sse2_done:
	RET

// func Grad1DStencilF32SSE2Asm(out, left, right *float32, invTwoDx float32, n int)
TEXT ·Grad1DStencilF32SSE2Asm(SB), NOSPLIT, $0-40
	MOVQ out+0(FP), R8
	MOVQ left+8(FP), R9
	MOVQ right+16(FP), R10
	MOVSS invTwoDx+24(FP), X7

	MOVQ n+32(FP), CX
	TESTQ CX, CX
	JZ   grad_sse2_done

grad_sse2_w4:
	CMPQ CX, $4
	JL   grad_sse2_tail

	VMOVUPS (R10), X0
	VMOVUPS (R9), X1
	VSUBPS  X1, X0, X0
	SHUFPS  $0, X7, X7
	MULPS   X7, X0
	VMOVUPS X0, (R8)

	ADDQ $16, R8
	ADDQ $16, R9
	ADDQ $16, R10
	SUBQ $4, CX
	JMP  grad_sse2_w4

grad_sse2_tail:
	TESTQ CX, CX
	JZ   grad_sse2_done

grad_sse2_scalar:
	MOVSS (R10), X0
	MOVSS (R9), X1
	SUBSS X1, X0
	MULSS X7, X0
	MOVSS X0, (R8)
	ADDQ  $4, R8
	ADDQ  $4, R9
	ADDQ  $4, R10
	DECQ  CX
	JNZ  grad_sse2_scalar

grad_sse2_done:
	RET

// func Laplacian4StencilF32SSE2Asm(out, um2, um1, u0, up1, up2 *float32, invDen float32, n int)
TEXT ·Laplacian4StencilF32SSE2Asm(SB), NOSPLIT, $0-64
	MOVQ out+0(FP), R8
	MOVQ um2+8(FP), R9
	MOVQ um1+16(FP), R10
	MOVQ u0+24(FP), R11
	MOVQ up1+32(FP), R12
	MOVQ up2+40(FP), R13
	MOVSS invDen+48(FP), X7

	MOVQ n+56(FP), CX
	TESTQ CX, CX
	JZ   lap4_sse2_done

lap4_sse2_w4:
	CMPQ CX, $4
	JL   lap4_sse2_tail

	VMOVUPS (R10), X0
	MOVSS phySse2C16<>(SB), X1
	SHUFPS $0, X1, X1
	MULPS X1, X0
	VMOVUPS (R11), X1
	MOVSS phySse2Cm30<>(SB), X2
	SHUFPS $0, X2, X2
	MULPS X2, X1
	ADDPS X1, X0
	VMOVUPS (R12), X1
	MOVSS phySse2C16<>(SB), X2
	SHUFPS $0, X2, X2
	MULPS X2, X1
	ADDPS X1, X0
	VMOVUPS (R9), X1
	MOVSS phySse2Cm1<>(SB), X2
	SHUFPS $0, X2, X2
	MULPS X2, X1
	ADDPS X1, X0
	VMOVUPS (R13), X1
	MOVSS phySse2Cm1<>(SB), X2
	SHUFPS $0, X2, X2
	MULPS X2, X1
	ADDPS X1, X0
	SHUFPS $0, X7, X7
	MULPS X7, X0
	VMOVUPS X0, (R8)

	ADDQ $16, R8
	ADDQ $16, R9
	ADDQ $16, R10
	ADDQ $16, R11
	ADDQ $16, R12
	ADDQ $16, R13
	SUBQ $4, CX
	JMP  lap4_sse2_w4

lap4_sse2_tail:
	TESTQ CX, CX
	JZ   lap4_sse2_done

lap4_sse2_scalar:
	MOVSS (R10), X0
	MOVSS phySse2C16<>(SB), X1
	MULSS X1, X0
	MOVSS (R11), X1
	MOVSS phySse2Cm30<>(SB), X2
	MULSS X2, X1
	ADDSS X1, X0
	MOVSS (R12), X1
	MOVSS phySse2C16<>(SB), X2
	MULSS X2, X1
	ADDSS X1, X0
	MOVSS (R9), X1
	MOVSS phySse2Cm1<>(SB), X2
	MULSS X2, X1
	ADDSS X1, X0
	MOVSS (R13), X1
	MOVSS phySse2Cm1<>(SB), X2
	MULSS X2, X1
	ADDSS X1, X0
	MULSS X7, X0
	MOVSS X0, (R8)
	ADDQ  $4, R8
	ADDQ  $4, R9
	ADDQ  $4, R10
	ADDQ  $4, R11
	ADDQ  $4, R12
	ADDQ  $4, R13
	DECQ  CX
	JNZ  lap4_sse2_scalar

lap4_sse2_done:
	RET
