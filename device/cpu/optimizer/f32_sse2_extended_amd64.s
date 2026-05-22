#include "textflag.h"

DATA optOneF32SSE2Ext<>+0(SB)/4, $0x3F800000
DATA optOneF32SSE2Ext<>+4(SB)/4, $0x3F800000
DATA optOneF32SSE2Ext<>+8(SB)/4, $0x3F800000
DATA optOneF32SSE2Ext<>+12(SB)/4, $0x3F800000
GLOBL optOneF32SSE2Ext<>(SB), RODATA|NOPTR, $16

DATA optNegOneF32SSE2Ext<>+0(SB)/4, $0xBF800000
DATA optNegOneF32SSE2Ext<>+4(SB)/4, $0xBF800000
DATA optNegOneF32SSE2Ext<>+8(SB)/4, $0xBF800000
DATA optNegOneF32SSE2Ext<>+12(SB)/4, $0xBF800000
GLOBL optNegOneF32SSE2Ext<>(SB), RODATA|NOPTR, $16

DATA optAbsMaskSSE2Ext<>+0(SB)/4, $0x7FFFFFFF
DATA optAbsMaskSSE2Ext<>+4(SB)/4, $0x7FFFFFFF
DATA optAbsMaskSSE2Ext<>+8(SB)/4, $0x7FFFFFFF
DATA optAbsMaskSSE2Ext<>+12(SB)/4, $0x7FFFFFFF
GLOBL optAbsMaskSSE2Ext<>(SB), RODATA|NOPTR, $16

// func AdamaxStepFloat32SSE2Asm(params, grad, first, infinity, output *float32, n int,
//     lr, beta1, beta2, eps, beta1Corr float32)
TEXT ·AdamaxStepFloat32SSE2Asm(SB), NOSPLIT, $0-68
	MOVQ params+0(FP), DI
	MOVQ grad+8(FP), SI
	MOVQ first+16(FP), R8
	MOVQ infinity+24(FP), R9
	MOVQ output+32(FP), R10
	MOVQ n+40(FP), CX

	MOVSS lr+48(FP), X6
	VBROADCASTSS X6, X6
	MOVSS beta1+52(FP), X7
	VBROADCASTSS X7, X7
	MOVSS beta2+56(FP), X8
	VBROADCASTSS X8, X8
	MOVSS eps+60(FP), X9
	VBROADCASTSS X9, X9
	MOVSS beta1Corr+64(FP), X10
	VBROADCASTSS X10, X10
	MOVSS optOneF32SSE2Ext<>(SB), X11
	VBROADCASTSS X11, X11
	VSUBPS X7, X11, X12
	MOVAPS optAbsMaskSSE2Ext<>(SB), X13

adamax_sse2_w4:
	CMPQ CX, $4
	JL   adamax_sse2_tail

	VMOVUPS (DI), X0
	VMOVUPS (SI), X1
	VMOVUPS (R8), X2
	VMOVUPS (R9), X3
	VMULPS X7, X2, X2
	VFMADD132PS X12, X1, X2
	VMULPS X8, X3, X3
	VMOVAPS X1, X4
	VANDPS X13, X4, X4
	VMAXPS X4, X3, X3
	VDIVPS X10, X2, X5
	VADDPS X9, X3, X14
	VMULPS X6, X5, X5
	VDIVPS X14, X5, X5
	VSUBPS X5, X0, X0
	VMOVUPS X2, (R8)
	VMOVUPS X3, (R9)
	VMOVUPS X0, (R10)

	ADDQ $16, DI
	ADDQ $16, SI
	ADDQ $16, R8
	ADDQ $16, R9
	ADDQ $16, R10
	SUBQ $4, CX
	JMP  adamax_sse2_w4

adamax_sse2_tail:
	TESTQ CX, CX
	JZ   adamax_sse2_done

adamax_sse2_scalar:
	VMOVSS (DI), X0
	VMOVSS (SI), X1
	VMOVSS (R8), X2
	VMOVSS (R9), X3
	VMULSS X7, X2, X2
	VFMADD132SS X12, X1, X2
	VMULSS X8, X3, X3
	VMOVAPS X1, X4
	VANDPS X13, X4, X4
	VMAXSS X4, X3, X3
	VDIVSS X10, X2, X5
	VADDSS X9, X3, X14
	VMULSS X6, X5, X5
	VDIVSS X14, X5, X5
	VSUBSS X5, X0, X0
	MOVSS X0, (R10)
	MOVSS X2, (R8)
	MOVSS X3, (R9)
	ADDQ $4, DI
	ADDQ $4, SI
	ADDQ $4, R8
	ADDQ $4, R9
	ADDQ $4, R10
	DECQ CX
	JNZ  adamax_sse2_scalar

adamax_sse2_done:
	RET

// func AdagradStepFloat32SSE2Asm(params, grad, accum, output *float32, n int, lr, eps float32)
TEXT ·AdagradStepFloat32SSE2Asm(SB), NOSPLIT, $0-48
	MOVQ params+0(FP), DI
	MOVQ grad+8(FP), SI
	MOVQ accum+16(FP), R8
	MOVQ output+24(FP), R9
	MOVQ n+32(FP), CX

	MOVSS lr+40(FP), X6
	VBROADCASTSS X6, X6
	MOVSS eps+44(FP), X9
	VBROADCASTSS X9, X9

ag_sse2_w4:
	CMPQ CX, $4
	JL   ag_sse2_tail

	VMOVUPS (DI), X0
	VMOVUPS (SI), X1
	VMOVUPS (R8), X2
	VFMADD213PS X1, X1, X2
	VSQRTPS X2, X4
	VADDPS X9, X4, X4
	VMULPS X6, X1, X5
	VDIVPS X4, X5, X5
	VSUBPS X5, X0, X0
	VMOVUPS X2, (R8)
	VMOVUPS X0, (R9)

	ADDQ $16, DI
	ADDQ $16, SI
	ADDQ $16, R8
	ADDQ $16, R9
	SUBQ $4, CX
	JMP  ag_sse2_w4

ag_sse2_tail:
	TESTQ CX, CX
	JZ   ag_sse2_done

ag_sse2_scalar:
	VMOVSS (DI), X0
	VMOVSS (SI), X1
	VMOVSS (R8), X2
	VFMADD213SS X1, X1, X2
	VSQRTSS X2, X2, X2
	VADDSS X9, X2, X4
	VMULSS X6, X1, X5
	VDIVSS X4, X5, X5
	VSUBSS X5, X0, X0
	MOVSS X0, (R9)
	MOVSS X2, (R8)
	ADDQ $4, DI
	ADDQ $4, SI
	ADDQ $4, R8
	ADDQ $4, R9
	DECQ CX
	JNZ  ag_sse2_scalar

ag_sse2_done:
	RET

// func RmspropStepFloat32SSE2Asm(params, grad, second, output *float32, n int, lr, decay, eps float32)
TEXT ·RmspropStepFloat32SSE2Asm(SB), NOSPLIT, $0-52
	MOVQ params+0(FP), DI
	MOVQ grad+8(FP), SI
	MOVQ second+16(FP), R8
	MOVQ output+24(FP), R9
	MOVQ n+32(FP), CX

	MOVSS lr+40(FP), X6
	VBROADCASTSS X6, X6
	MOVSS decay+44(FP), X7
	VBROADCASTSS X7, X7
	MOVSS eps+48(FP), X9
	VBROADCASTSS X9, X9
	MOVSS optOneF32SSE2Ext<>(SB), X10
	VBROADCASTSS X10, X10
	VSUBPS X7, X10, X11

rms_sse2_w4:
	CMPQ CX, $4
	JL   rms_sse2_tail

	VMOVUPS (DI), X0
	VMOVUPS (SI), X1
	VMOVUPS (R8), X2
	VMULPS X1, X1, X4
	VMULPS X11, X4, X4
	VFMADD213PS X7, X2, X4
	VSQRTPS X4, X5
	VADDPS X9, X5, X5
	VMULPS X6, X1, X6
	VDIVPS X5, X6, X6
	VSUBPS X6, X0, X0
	VMOVUPS X4, (R8)
	VMOVUPS X0, (R9)

	ADDQ $16, DI
	ADDQ $16, SI
	ADDQ $16, R8
	ADDQ $16, R9
	SUBQ $4, CX
	JMP  rms_sse2_w4

rms_sse2_tail:
	TESTQ CX, CX
	JZ   rms_sse2_done

rms_sse2_scalar:
	VMOVSS (DI), X0
	VMOVSS (SI), X1
	VMOVSS (R8), X2
	VMULSS X1, X1, X4
	VMULSS X11, X4, X4
	VFMADD213SS X7, X2, X4
	VSQRTSS X4, X4, X4
	VADDSS X9, X4, X5
	VMULSS X6, X1, X6
	VDIVSS X5, X6, X6
	VSUBSS X6, X0, X0
	MOVSS X0, (R9)
	MOVSS X4, (R8)
	ADDQ $4, DI
	ADDQ $4, SI
	ADDQ $4, R8
	ADDQ $4, R9
	DECQ CX
	JNZ  rms_sse2_scalar

rms_sse2_done:
	RET

// func LionStepFloat32SSE2Asm(params, grad, momentum, output *float32, n int,
//     lr, beta1, beta2, weightDecay float32)
TEXT ·LionStepFloat32SSE2Asm(SB), NOSPLIT, $0-56
	MOVQ params+0(FP), DI
	MOVQ grad+8(FP), SI
	MOVQ momentum+16(FP), R8
	MOVQ output+24(FP), R9
	MOVQ n+32(FP), CX

	MOVSS lr+40(FP), X6
	VBROADCASTSS X6, X6
	MOVSS beta1+44(FP), X7
	VBROADCASTSS X7, X7
	MOVSS beta2+48(FP), X8
	VBROADCASTSS X8, X8
	MOVSS weightDecay+52(FP), X9
	VBROADCASTSS X9, X9
	MOVSS optOneF32SSE2Ext<>(SB), X10
	VBROADCASTSS X10, X10
	VSUBPS X7, X10, X11
	VSUBPS X8, X10, X12
	MOVAPS optNegOneF32SSE2Ext<>(SB), X13

lion_sse2_w4:
	CMPQ CX, $4
	JL   lion_sse2_tail

	VMOVUPS (DI), X0
	VMOVUPS (SI), X1
	VMOVUPS (R8), X2
	VMULPS X11, X1, X3
	VFMADD213PS X7, X2, X3
	XORPS X4, X4
	VCMPPS $6, X3, X2, X14
	MOVAPS optOneF32SSE2Ext<>(SB), X15
	VANDPS X14, X15, X5
	VCMPPS $1, X3, X2, X14
	MOVAPS optNegOneF32SSE2Ext<>(SB), X13
	VANDPS X14, X13, X14
	VORPS X14, X5, X4
	VFMADD213PS X9, X0, X4
	VMULPS X6, X4, X4
	VSUBPS X4, X0, X0
	VMULPS X12, X1, X5
	VFMADD213PS X8, X2, X5
	VMOVUPS X5, (R8)
	VMOVUPS X0, (R9)

	ADDQ $16, DI
	ADDQ $16, SI
	ADDQ $16, R8
	ADDQ $16, R9
	SUBQ $4, CX
	JMP  lion_sse2_w4

lion_sse2_tail:
	TESTQ CX, CX
	JZ   lion_sse2_done

lion_sse2_scalar:
	VMOVSS (DI), X0
	VMOVSS (SI), X1
	VMOVSS (R8), X2
	VMULSS X11, X1, X3
	VFMADD213SS X7, X2, X3
	XORPS X4, X4
	UCOMISS X3, X2
	JBE  lion_sse2_scalar_not_gt
	MOVSS optOneF32SSE2Ext<>(SB), X4
lion_sse2_scalar_not_gt:
	UCOMISS X2, X3
	JBE  lion_sse2_scalar_not_lt
	MOVSS optNegOneF32SSE2Ext<>(SB), X4
lion_sse2_scalar_not_lt:
	VFMADD213SS X9, X0, X4
	VMULSS X6, X4, X4
	VSUBSS X4, X0, X0
	VMULSS X12, X1, X5
	VFMADD213SS X8, X2, X5
	MOVSS X5, (R8)
	MOVSS X0, (R9)
	ADDQ $4, DI
	ADDQ $4, SI
	ADDQ $4, R8
	ADDQ $4, R9
	DECQ CX
	JNZ  lion_sse2_scalar

lion_sse2_done:
	RET

// func LbfgsStepFloat32SSE2Asm(params, grad, output *float32, n int, lr float32)
TEXT ·LbfgsStepFloat32SSE2Asm(SB), NOSPLIT, $0-36
	MOVQ params+0(FP), DI
	MOVQ grad+8(FP), SI
	MOVQ output+16(FP), R8
	MOVQ n+24(FP), CX
	MOVSS lr+32(FP), X6
	VBROADCASTSS X6, X6

lbfgs_sse2_w4:
	CMPQ CX, $4
	JL   lbfgs_sse2_tail

	VMOVUPS (DI), X0
	VMOVUPS (SI), X1
	VMULPS X6, X1, X1
	VSUBPS X1, X0, X0
	VMOVUPS X0, (R8)

	ADDQ $16, DI
	ADDQ $16, SI
	ADDQ $16, R8
	SUBQ $4, CX
	JMP  lbfgs_sse2_w4

lbfgs_sse2_tail:
	TESTQ CX, CX
	JZ   lbfgs_sse2_done

lbfgs_sse2_scalar:
	VMOVSS (DI), X0
	VMOVSS (SI), X1
	VMULSS X6, X1, X1
	VSUBSS X1, X0, X0
	MOVSS X0, (R8)
	ADDQ $4, DI
	ADDQ $4, SI
	ADDQ $4, R8
	DECQ CX
	JNZ  lbfgs_sse2_scalar

lbfgs_sse2_done:
	RET

// func LarsStepFloat32SSE2Asm(params, grad, momentum, output *float32, n int,
//     lr, momentumFactor, weightDecay, effectiveLr float32)
TEXT ·LarsStepFloat32SSE2Asm(SB), NOSPLIT, $0-56
	MOVQ params+0(FP), DI
	MOVQ grad+8(FP), SI
	MOVQ momentum+16(FP), R8
	MOVQ output+24(FP), R9
	MOVQ n+32(FP), CX

	MOVSS momentumFactor+44(FP), X7
	VBROADCASTSS X7, X7
	MOVSS weightDecay+48(FP), X8
	VBROADCASTSS X8, X8
	MOVSS effectiveLr+52(FP), X6
	VBROADCASTSS X6, X6
	XORPS X10, X10
	VSUBPS X6, X10, X10

lars_sse2_w4:
	CMPQ CX, $4
	JL   lars_sse2_tail

	VMOVUPS (DI), X0
	VMOVUPS (SI), X1
	VMOVUPS (R8), X2
	VFMADD213PS X8, X0, X1
	VFMADD213PS X7, X2, X1
	VFMADD213PS X10, X1, X0
	VMOVUPS X1, (R8)
	VMOVUPS X0, (R9)

	ADDQ $16, DI
	ADDQ $16, SI
	ADDQ $16, R8
	ADDQ $16, R9
	SUBQ $4, CX
	JMP  lars_sse2_w4

lars_sse2_tail:
	TESTQ CX, CX
	JZ   lars_sse2_done

lars_sse2_scalar:
	VMOVSS (DI), X0
	VMOVSS (SI), X1
	VMOVSS (R8), X2
	VFMADD213SS X8, X0, X1
	VFMADD213SS X7, X2, X1
	VFMADD213SS X10, X1, X0
	MOVSS X1, (R8)
	MOVSS X0, (R9)
	ADDQ $4, DI
	ADDQ $4, SI
	ADDQ $4, R8
	ADDQ $4, R9
	DECQ CX
	JNZ  lars_sse2_scalar

lars_sse2_done:
	RET

// func HebbianStepRowFloat32SSE2Asm(weights, pre, output *float32, n int, decayFactor, lrPost float32)
TEXT ·HebbianStepRowFloat32SSE2Asm(SB), NOSPLIT, $0-40
	MOVQ weights+0(FP), DI
	MOVQ pre+8(FP), SI
	MOVQ output+16(FP), R8
	MOVQ n+24(FP), CX
	MOVSS decayFactor+32(FP), X6
	VBROADCASTSS X6, X6
	MOVSS lrPost+36(FP), X7
	VBROADCASTSS X7, X7

hebb_sse2_w4:
	CMPQ CX, $4
	JL   hebb_sse2_tail

	VMOVUPS (DI), X0
	VMOVUPS (SI), X1
	VMULPS X6, X0, X3
	VFMADD213PS X7, X1, X3
	VMOVUPS X3, (R8)

	ADDQ $16, DI
	ADDQ $16, SI
	ADDQ $16, R8
	SUBQ $4, CX
	JMP  hebb_sse2_w4

hebb_sse2_tail:
	TESTQ CX, CX
	JZ   hebb_sse2_done

hebb_sse2_scalar:
	VMOVSS (DI), X0
	VMOVSS (SI), X1
	VMULSS X6, X0, X3
	VFMADD213SS X7, X1, X3
	MOVSS X3, (R8)
	ADDQ $4, DI
	ADDQ $4, SI
	ADDQ $4, R8
	DECQ CX
	JNZ  hebb_sse2_scalar

hebb_sse2_done:
	RET
