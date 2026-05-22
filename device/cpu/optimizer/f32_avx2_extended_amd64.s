#include "textflag.h"

// func AdamaxStepFloat32AVX2Asm(params, grad, first, infinity, output *float32, n int,
//     lr, beta1, beta2, eps, beta1Corr float32)
TEXT ·AdamaxStepFloat32AVX2Asm(SB), NOSPLIT, $0-68
	MOVQ params+0(FP), DI
	MOVQ grad+8(FP), SI
	MOVQ first+16(FP), R8
	MOVQ infinity+24(FP), R9
	MOVQ output+32(FP), R10
	MOVQ n+40(FP), CX

	MOVSS lr+48(FP), X6
	VBROADCASTSS X6, Y8
	MOVSS beta1+52(FP), X7
	VBROADCASTSS X7, Y9
	MOVSS beta2+56(FP), X8
	VBROADCASTSS X8, Y10
	MOVSS eps+60(FP), X9
	VBROADCASTSS X9, Y11
	MOVSS beta1Corr+64(FP), X10
	VBROADCASTSS X10, Y12
	VBROADCASTSS optOneF32AVX2<>(SB), Y14
	VSUBPS Y9, Y14, Y15
	VBROADCASTSS optAbsMaskAVX2<>(SB), Y13

adamax_avx2_w8:
	CMPQ CX, $8
	JL   adamax_avx2_w4

	VMOVUPS (DI), Y0
	VMOVUPS (SI), Y1
	VMOVUPS (R8), Y2
	VMOVUPS (R9), Y3
	VMULPS Y9, Y2, Y2
	VFMADD132PS Y15, Y1, Y2
	VMULPS Y10, Y3, Y3
	VMOVAPS Y1, Y4
	VANDPS Y13, Y4, Y4
	VMAXPS Y4, Y3, Y3
	VDIVPS Y12, Y2, Y5
	VADDPS Y11, Y3, Y6
	VMULPS Y8, Y5, Y5
	VDIVPS Y6, Y5, Y5
	VSUBPS Y5, Y0, Y0
	VMOVUPS Y2, (R8)
	VMOVUPS Y3, (R9)
	VMOVUPS Y0, (R10)

	ADDQ $32, DI
	ADDQ $32, SI
	ADDQ $32, R8
	ADDQ $32, R9
	ADDQ $32, R10
	SUBQ $8, CX
	JMP  adamax_avx2_w8

adamax_avx2_w4:
	CMPQ CX, $4
	JL   adamax_avx2_tail

	VMOVUPS (DI), X0
	VMOVUPS (SI), X1
	VMOVUPS (R8), X2
	VMOVUPS (R9), X3
	VMULPS X9, X2, X2
	VFMADD132PS X15, X1, X2
	VMULPS X10, X3, X3
	VMOVAPS X1, X4
	VANDPS X13, X4, X4
	VMAXPS X4, X3, X3
	VDIVPS X12, X2, X6
	VADDPS X11, X3, X7
	VMULPS X8, X6, X6
	VDIVPS X7, X6, X6
	VSUBPS X6, X0, X0
	VMOVUPS X2, (R8)
	VMOVUPS X3, (R9)
	VMOVUPS X0, (R10)

	ADDQ $16, DI
	ADDQ $16, SI
	ADDQ $16, R8
	ADDQ $16, R9
	ADDQ $16, R10
	SUBQ $4, CX
	JMP  adamax_avx2_w4

adamax_avx2_tail:
	TESTQ CX, CX
	JZ   adamax_avx2_done

adamax_avx2_scalar:
	VMOVSS (DI), X0
	VMOVSS (SI), X1
	VMOVSS (R8), X2
	VMOVSS (R9), X3
	VMULSS X9, X2, X2
	VFMADD132SS X15, X1, X2
	VMULSS X10, X3, X3
	VMOVAPS X1, X4
	VANDPS X13, X4, X4
	VMAXSS X4, X3, X3
	VDIVSS X12, X2, X5
	VADDSS X11, X3, X6
	VMULSS X8, X5, X5
	VDIVSS X6, X5, X5
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
	JNZ  adamax_avx2_scalar

adamax_avx2_done:
	RET

// func AdagradStepFloat32AVX2Asm(params, grad, accum, output *float32, n int, lr, eps float32)
TEXT ·AdagradStepFloat32AVX2Asm(SB), NOSPLIT, $0-48
	MOVQ params+0(FP), DI
	MOVQ grad+8(FP), SI
	MOVQ accum+16(FP), R8
	MOVQ output+24(FP), R9
	MOVQ n+32(FP), CX

	MOVSS lr+40(FP), X6
	VBROADCASTSS X6, Y8
	MOVSS eps+44(FP), X9
	VBROADCASTSS X9, Y11

ag_avx2_w8:
	CMPQ CX, $8
	JL   ag_avx2_w4

	VMOVUPS (DI), Y0
	VMOVUPS (SI), Y1
	VMOVUPS (R8), Y2
	VFMADD213PS Y1, Y1, Y2
	VSQRTPS Y2, Y4
	VADDPS Y11, Y4, Y4
	VMULPS Y8, Y1, Y5
	VDIVPS Y4, Y5, Y5
	VSUBPS Y5, Y0, Y0
	VMOVUPS Y2, (R8)
	VMOVUPS Y0, (R9)

	ADDQ $32, DI
	ADDQ $32, SI
	ADDQ $32, R8
	ADDQ $32, R9
	SUBQ $8, CX
	JMP  ag_avx2_w8

ag_avx2_w4:
	CMPQ CX, $4
	JL   ag_avx2_tail

	VMOVUPS (DI), X0
	VMOVUPS (SI), X1
	VMOVUPS (R8), X2
	VFMADD213PS X1, X1, X2
	VSQRTPS X2, X4
	VADDPS X11, X4, X4
	VMULPS X8, X1, X5
	VDIVPS X4, X5, X5
	VSUBPS X5, X0, X0
	VMOVUPS X2, (R8)
	VMOVUPS X0, (R9)

	ADDQ $16, DI
	ADDQ $16, SI
	ADDQ $16, R8
	ADDQ $16, R9
	SUBQ $4, CX
	JMP  ag_avx2_w4

ag_avx2_tail:
	TESTQ CX, CX
	JZ   ag_avx2_done

ag_avx2_scalar:
	VMOVSS (DI), X0
	VMOVSS (SI), X1
	VMOVSS (R8), X2
	VFMADD213SS X1, X1, X2
	VSQRTSS X2, X2, X2
	VADDSS X11, X2, X4
	VMULSS X8, X1, X5
	VDIVSS X4, X5, X5
	VSUBSS X5, X0, X0
	MOVSS X0, (R9)
	MOVSS X2, (R8)
	ADDQ $4, DI
	ADDQ $4, SI
	ADDQ $4, R8
	ADDQ $4, R9
	DECQ CX
	JNZ  ag_avx2_scalar

ag_avx2_done:
	RET

// func RmspropStepFloat32AVX2Asm(params, grad, second, output *float32, n int, lr, decay, eps float32)
TEXT ·RmspropStepFloat32AVX2Asm(SB), NOSPLIT, $0-52
	MOVQ params+0(FP), DI
	MOVQ grad+8(FP), SI
	MOVQ second+16(FP), R8
	MOVQ output+24(FP), R9
	MOVQ n+32(FP), CX

	MOVSS lr+40(FP), X6
	VBROADCASTSS X6, Y8
	MOVSS decay+44(FP), X7
	VBROADCASTSS X7, Y9
	MOVSS eps+48(FP), X9
	VBROADCASTSS X9, Y11
	VBROADCASTSS optOneF32AVX2<>(SB), Y14
	VSUBPS Y9, Y14, Y15

rms_avx2_w8:
	CMPQ CX, $8
	JL   rms_avx2_w4

	VMOVUPS (DI), Y0
	VMOVUPS (SI), Y1
	VMOVUPS (R8), Y2
	VMULPS Y1, Y1, Y4
	VMULPS Y15, Y4, Y4
	VFMADD213PS Y9, Y2, Y4
	VSQRTPS Y4, Y5
	VADDPS Y11, Y5, Y5
	VMULPS Y8, Y1, Y6
	VDIVPS Y5, Y6, Y6
	VSUBPS Y6, Y0, Y0
	VMOVUPS Y4, (R8)
	VMOVUPS Y0, (R9)

	ADDQ $32, DI
	ADDQ $32, SI
	ADDQ $32, R8
	ADDQ $32, R9
	SUBQ $8, CX
	JMP  rms_avx2_w8

rms_avx2_w4:
	CMPQ CX, $4
	JL   rms_avx2_tail

	VMOVUPS (DI), X0
	VMOVUPS (SI), X1
	VMOVUPS (R8), X2
	VMULPS X1, X1, X4
	VMULPS X15, X4, X4
	VFMADD213PS X9, X2, X4
	VSQRTPS X4, X5
	VADDPS X11, X5, X5
	VMULPS X8, X1, X6
	VDIVPS X5, X6, X6
	VSUBPS X6, X0, X0
	VMOVUPS X4, (R8)
	VMOVUPS X0, (R9)

	ADDQ $16, DI
	ADDQ $16, SI
	ADDQ $16, R8
	ADDQ $16, R9
	SUBQ $4, CX
	JMP  rms_avx2_w4

rms_avx2_tail:
	TESTQ CX, CX
	JZ   rms_avx2_done

rms_avx2_scalar:
	VMOVSS (DI), X0
	VMOVSS (SI), X1
	VMOVSS (R8), X2
	VMULSS X1, X1, X4
	VMULSS X15, X4, X4
	VFMADD213SS X9, X2, X4
	VSQRTSS X4, X4, X4
	VADDSS X11, X4, X5
	VMULSS X8, X1, X6
	VDIVSS X5, X6, X6
	VSUBSS X6, X0, X0
	MOVSS X0, (R9)
	MOVSS X4, (R8)
	ADDQ $4, DI
	ADDQ $4, SI
	ADDQ $4, R8
	ADDQ $4, R9
	DECQ CX
	JNZ  rms_avx2_scalar

rms_avx2_done:
	RET

// func LionStepFloat32AVX2Asm(params, grad, momentum, output *float32, n int,
//     lr, beta1, beta2, weightDecay float32)
TEXT ·LionStepFloat32AVX2Asm(SB), NOSPLIT, $0-56
	MOVQ params+0(FP), DI
	MOVQ grad+8(FP), SI
	MOVQ momentum+16(FP), R8
	MOVQ output+24(FP), R9
	MOVQ n+32(FP), CX

	MOVSS lr+40(FP), X6
	VBROADCASTSS X6, Y6
	MOVSS beta1+44(FP), X7
	VBROADCASTSS X7, Y9
	MOVSS beta2+48(FP), X8
	VBROADCASTSS X8, Y10
	MOVSS weightDecay+52(FP), X9
	VBROADCASTSS X9, Y11
	VBROADCASTSS optOneF32AVX2<>(SB), Y14
	VSUBPS Y9, Y14, Y13
	VSUBPS Y10, Y14, Y12
	VBROADCASTSS optNegOneF32AVX2<>(SB), Y15

lion_avx2_w8:
	CMPQ CX, $8
	JL   lion_avx2_w4

	VMOVUPS (DI), Y0
	VMOVUPS (SI), Y1
	VMOVUPS (R8), Y2
	VMULPS Y13, Y1, Y3
	VFMADD213PS Y9, Y2, Y3
	VXORPS Y7, Y7, Y7
	VCMPPS $1, Y3, Y2, Y4
	VBLENDVPS Y4, Y15, Y7, Y4
	VCMPPS $6, Y3, Y2, Y5
	VBLENDVPS Y5, Y14, Y4, Y5
	VMOVAPS Y5, Y8
	VFMADD213PS Y11, Y0, Y8
	VMULPS Y6, Y8, Y8
	VSUBPS Y8, Y0, Y0
	VMULPS Y12, Y1, Y4
	VFMADD213PS Y10, Y2, Y4
	VMOVUPS Y4, (R8)
	VMOVUPS Y0, (R9)

	ADDQ $32, DI
	ADDQ $32, SI
	ADDQ $32, R8
	ADDQ $32, R9
	SUBQ $8, CX
	JMP  lion_avx2_w8

lion_avx2_w4:
	CMPQ CX, $4
	JL   lion_avx2_tail

	VMOVUPS (DI), X0
	VMOVUPS (SI), X1
	VMOVUPS (R8), X2
	VMULPS X13, X1, X3
	VFMADD213PS X9, X2, X3
	VXORPS X7, X7, X7
	VCMPPS $1, X3, X2, X4
	VBLENDVPS X4, X15, X7, X4
	VCMPPS $6, X3, X2, X5
	VBLENDVPS X5, X14, X4, X5
	VMOVAPS X5, X8
	VFMADD213PS X11, X0, X8
	VMULPS X6, X8, X8
	VSUBPS X8, X0, X0
	VMULPS X12, X1, X4
	VFMADD213PS X10, X2, X4
	VMOVUPS X4, (R8)
	VMOVUPS X0, (R9)

	ADDQ $16, DI
	ADDQ $16, SI
	ADDQ $16, R8
	ADDQ $16, R9
	SUBQ $4, CX
	JMP  lion_avx2_w4

lion_avx2_tail:
	TESTQ CX, CX
	JZ   lion_avx2_done

lion_avx2_scalar:
	VMOVSS (DI), X0
	VMOVSS (SI), X1
	VMOVSS (R8), X2
	VMULSS X13, X1, X3
	VFMADD213SS X9, X2, X3
	XORPS X8, X8
	UCOMISS X3, X2
	JBE  lion_avx2_scalar_not_gt
	MOVSS optOneF32AVX2<>(SB), X8
lion_avx2_scalar_not_gt:
	UCOMISS X2, X3
	JBE  lion_avx2_scalar_not_lt
	MOVSS optNegOneF32AVX2<>(SB), X8
lion_avx2_scalar_not_lt:
	VFMADD213SS X11, X0, X8
	VMULSS X6, X8, X8
	VSUBSS X8, X0, X0
	VMULSS X12, X1, X4
	VFMADD213SS X10, X2, X4
	MOVSS X4, (R8)
	MOVSS X0, (R9)
	ADDQ $4, DI
	ADDQ $4, SI
	ADDQ $4, R8
	ADDQ $4, R9
	DECQ CX
	JNZ  lion_avx2_scalar

lion_avx2_done:
	RET

// func LbfgsStepFloat32AVX2Asm(params, grad, output *float32, n int, lr float32)
TEXT ·LbfgsStepFloat32AVX2Asm(SB), NOSPLIT, $0-36
	MOVQ params+0(FP), DI
	MOVQ grad+8(FP), SI
	MOVQ output+16(FP), R8
	MOVQ n+24(FP), CX
	MOVSS lr+32(FP), X6
	VBROADCASTSS X6, Y8

lbfgs_avx2_w8:
	CMPQ CX, $8
	JL   lbfgs_avx2_w4

	VMOVUPS (DI), Y0
	VMOVUPS (SI), Y1
	VMULPS Y8, Y1, Y1
	VSUBPS Y1, Y0, Y0
	VMOVUPS Y0, (R8)

	ADDQ $32, DI
	ADDQ $32, SI
	ADDQ $32, R8
	SUBQ $8, CX
	JMP  lbfgs_avx2_w8

lbfgs_avx2_w4:
	CMPQ CX, $4
	JL   lbfgs_avx2_tail

	VMOVUPS (DI), X0
	VMOVUPS (SI), X1
	VMULPS X8, X1, X1
	VSUBPS X1, X0, X0
	VMOVUPS X0, (R8)

	ADDQ $16, DI
	ADDQ $16, SI
	ADDQ $16, R8
	SUBQ $4, CX
	JMP  lbfgs_avx2_w4

lbfgs_avx2_tail:
	TESTQ CX, CX
	JZ   lbfgs_avx2_done

lbfgs_avx2_scalar:
	VMOVSS (DI), X0
	VMOVSS (SI), X1
	VMULSS X8, X1, X1
	VSUBSS X1, X0, X0
	MOVSS X0, (R8)
	ADDQ $4, DI
	ADDQ $4, SI
	ADDQ $4, R8
	DECQ CX
	JNZ  lbfgs_avx2_scalar

lbfgs_avx2_done:
	RET

// func LarsStepFloat32AVX2Asm(params, grad, momentum, output *float32, n int,
//     lr, momentumFactor, weightDecay, effectiveLr float32)
TEXT ·LarsStepFloat32AVX2Asm(SB), NOSPLIT, $0-56
	MOVQ params+0(FP), DI
	MOVQ grad+8(FP), SI
	MOVQ momentum+16(FP), R8
	MOVQ output+24(FP), R9
	MOVQ n+32(FP), CX

	MOVSS momentumFactor+44(FP), X7
	VBROADCASTSS X7, Y9
	MOVSS weightDecay+48(FP), X8
	VBROADCASTSS X8, Y10
	MOVSS effectiveLr+52(FP), X6
	VBROADCASTSS X6, Y8
	VXORPS Y7, Y7, Y7
	VSUBPS Y8, Y7, Y7

lars_avx2_w8:
	CMPQ CX, $8
	JL   lars_avx2_w4

	VMOVUPS (DI), Y0
	VMOVUPS (SI), Y1
	VMOVUPS (R8), Y2
	VFMADD213PS Y10, Y0, Y1
	VFMADD213PS Y9, Y2, Y1
	VFMADD213PS Y7, Y1, Y0
	VMOVUPS Y1, (R8)
	VMOVUPS Y0, (R9)

	ADDQ $32, DI
	ADDQ $32, SI
	ADDQ $32, R8
	ADDQ $32, R9
	SUBQ $8, CX
	JMP  lars_avx2_w8

lars_avx2_w4:
	CMPQ CX, $4
	JL   lars_avx2_tail

	VMOVUPS (DI), X0
	VMOVUPS (SI), X1
	VMOVUPS (R8), X2
	VFMADD213PS X10, X0, X1
	VFMADD213PS X9, X2, X1
	VFMADD213PS X7, X1, X0
	VMOVUPS X1, (R8)
	VMOVUPS X0, (R9)

	ADDQ $16, DI
	ADDQ $16, SI
	ADDQ $16, R8
	ADDQ $16, R9
	SUBQ $4, CX
	JMP  lars_avx2_w4

lars_avx2_tail:
	TESTQ CX, CX
	JZ   lars_avx2_done

lars_avx2_scalar:
	VMOVSS (DI), X0
	VMOVSS (SI), X1
	VMOVSS (R8), X2
	VFMADD213SS X10, X0, X1
	VFMADD213SS X9, X2, X1
	VFMADD213SS X7, X1, X0
	MOVSS X1, (R8)
	MOVSS X0, (R9)
	ADDQ $4, DI
	ADDQ $4, SI
	ADDQ $4, R8
	ADDQ $4, R9
	DECQ CX
	JNZ  lars_avx2_scalar

lars_avx2_done:
	RET

// func HebbianStepRowFloat32AVX2Asm(weights, pre, output *float32, n int, decayFactor, lrPost float32)
TEXT ·HebbianStepRowFloat32AVX2Asm(SB), NOSPLIT, $0-40
	MOVQ weights+0(FP), DI
	MOVQ pre+8(FP), SI
	MOVQ output+16(FP), R8
	MOVQ n+24(FP), CX
	MOVSS decayFactor+32(FP), X6
	VBROADCASTSS X6, Y8
	MOVSS lrPost+36(FP), X7
	VBROADCASTSS X7, Y9

hebb_avx2_w8:
	CMPQ CX, $8
	JL   hebb_avx2_w4

	VMOVUPS (DI), Y0
	VMOVUPS (SI), Y1
	VMULPS Y8, Y0, Y3
	VFMADD213PS Y9, Y1, Y3
	VMOVUPS Y3, (R8)

	ADDQ $32, DI
	ADDQ $32, SI
	ADDQ $32, R8
	SUBQ $8, CX
	JMP  hebb_avx2_w8

hebb_avx2_w4:
	CMPQ CX, $4
	JL   hebb_avx2_tail

	VMOVUPS (DI), X0
	VMOVUPS (SI), X1
	VMULPS X8, X0, X3
	VFMADD213PS X9, X1, X3
	VMOVUPS X3, (R8)

	ADDQ $16, DI
	ADDQ $16, SI
	ADDQ $16, R8
	SUBQ $4, CX
	JMP  hebb_avx2_w4

hebb_avx2_tail:
	TESTQ CX, CX
	JZ   hebb_avx2_done

hebb_avx2_scalar:
	VMOVSS (DI), X0
	VMOVSS (SI), X1
	VMULSS X8, X0, X3
	VFMADD213SS X9, X1, X3
	MOVSS X3, (R8)
	ADDQ $4, DI
	ADDQ $4, SI
	ADDQ $4, R8
	DECQ CX
	JNZ  hebb_avx2_scalar

hebb_avx2_done:
	RET
