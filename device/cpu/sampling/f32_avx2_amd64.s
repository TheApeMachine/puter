#include "textflag.h"

DATA samAvx2OneF32<>+0(SB)/4, $0x3f800000
GLOBL samAvx2OneF32<>(SB), RODATA|NOPTR, $4

DATA samAvx2ExpC<>+0(SB)/4, $1.4426950408889634
DATA samAvx2ExpC<>+4(SB)/4, $0.6931471805599453
DATA samAvx2ExpC<>+12(SB)/4, $0.00019841270
DATA samAvx2ExpC<>+16(SB)/4, $0.0013888889
DATA samAvx2ExpC<>+20(SB)/4, $0.008333334
DATA samAvx2ExpC<>+24(SB)/4, $0.041666667
DATA samAvx2ExpC<>+28(SB)/4, $0.16666667
DATA samAvx2ExpC<>+32(SB)/4, $0.5
DATA samAvx2ExpC<>+36(SB)/4, $1.0
DATA samAvx2ExpC<>+40(SB)/4, $1.0
GLOBL samAvx2ExpC<>(SB), RODATA|NOPTR, $44

DATA samAvx2ExpBias127<>+0(SB)/4, $127
GLOBL samAvx2ExpBias127<>(SB), RODATA|NOPTR, $4

DATA samAvx2SoftmaxClamp<>+0(SB)/4, $-87.0
GLOBL samAvx2SoftmaxClamp<>(SB), RODATA|NOPTR, $4

// func GreedySampleFloat32AVX2Asm(logits *float32, count int) int32
TEXT ·GreedySampleFloat32AVX2Asm(SB), NOSPLIT, $0-20
	MOVQ logits+0(FP), SI
	MOVQ SI, BX
	MOVQ count+8(FP), CX

	TESTQ CX, CX
	JZ   greedy_avx2_zero

	CMPQ CX, $1
	JE   greedy_avx2_one

	MOVSS (SI), X0
	SHUFPS $0x00, X0, X0
	ADDQ $4, SI
	DECQ CX

greedy_avx2_max_w8:
	CMPQ CX, $8
	JL   greedy_avx2_max_w4

	MOVUPS (SI), X4
	MAXPS X4, X0
	MOVUPS 16(SI), X4
	MAXPS X4, X0

	ADDQ $32, SI
	SUBQ $8, CX
	JMP  greedy_avx2_max_w8

greedy_avx2_max_w4:
	CMPQ CX, $4
	JL   greedy_avx2_max_tail

	MOVUPS (SI), X4
	MAXPS X4, X0

	ADDQ $16, SI
	SUBQ $4, CX
	JMP  greedy_avx2_max_w4

greedy_avx2_max_tail:
	TESTQ CX, CX
	JZ   greedy_avx2_max_done

greedy_avx2_max_scalar:
	MOVSS (SI), X4
	MAXSS X4, X0
	ADDQ  $4, SI
	DECQ  CX
	JNZ  greedy_avx2_max_scalar

greedy_avx2_max_done:
	MOVAPS X0, X4
	SHUFPS $0x4E, X0, X4
	MAXPS  X4, X0
	MOVAPS X0, X4
	SHUFPS $0xB1, X0, X4
	MAXPS  X4, X0
	MOVSS  X0, X0

	MOVQ BX, SI
	MOVQ count+8(FP), CX
	XORQ R8, R8

greedy_avx2_find_scalar:
	CMPQ R8, CX
	JGE  greedy_avx2_fail

	MOVSS (SI), X4
	UCOMISS X0, X4
	JNE  greedy_avx2_find_next
	MOVL R8, ret+16(FP)
	RET

greedy_avx2_find_next:
	ADDQ $4, SI
	INCQ R8
	JMP  greedy_avx2_find_scalar

greedy_avx2_fail:
	MOVQ count+8(FP), AX
	DECQ AX
	MOVL AX, ret+16(FP)
	RET

greedy_avx2_one:
	XORL AX, AX
	MOVL AX, ret+16(FP)
	RET

greedy_avx2_zero:
	XORL AX, AX
	MOVL AX, ret+16(FP)
	RET

// func SamplingSoftmaxRowFloat32AVX2Asm(logits, out *float32, temperature float32, count int)
TEXT ·SamplingSoftmaxRowFloat32AVX2Asm(SB), NOSPLIT, $0-28
	MOVQ logits+0(FP), SI
	MOVQ out+8(FP), DI
	MOVSS temperature+16(FP), X10
	MOVQ count+24(FP), CX
	TESTQ CX, CX
	JZ   sam_avx2_softmax_done

	XORPS X11, X11
	UCOMISS X10, X11
	JNE  sam_avx2_softmax_temp_ok
	MOVSS samAvx2OneF32<>(SB), X10

sam_avx2_softmax_temp_ok:
	VBROADCASTSS X10, Y10

	MOVSS (SI), X0
	VBROADCASTSS X0, Y0
	ADDQ $4, SI
	DECQ CX

sam_avx2_softmax_max_w8:
	CMPQ CX, $8
	JL   sam_avx2_softmax_max_w4

	VMOVUPS (SI), Y1
	VMAXPS  Y1, Y0, Y0

	ADDQ $32, SI
	SUBQ $8, CX
	JMP  sam_avx2_softmax_max_w8

sam_avx2_softmax_max_w4:
	CMPQ CX, $4
	JL   sam_avx2_softmax_max_reduce

	VMOVUPS (SI), Y1
	VMAXPS  Y1, Y0, Y0

	ADDQ $16, SI
	SUBQ $4, CX
	JMP  sam_avx2_softmax_max_w4

sam_avx2_softmax_max_reduce:
	VEXTRACTF128 $0, Y0, X0
	VEXTRACTF128 $1, Y0, X1
	VMAXPS  X1, X0, X0
	VHADDPS X0, X0, X0
	VHADDPS X0, X0, X0

	TESTQ CX, CX
	JZ   sam_avx2_softmax_max_broadcast

sam_avx2_softmax_max_scalar:
	MOVSS (SI), X1
	MAXSS X1, X0
	ADDQ  $4, SI
	DECQ  CX
	JNZ  sam_avx2_softmax_max_scalar

sam_avx2_softmax_max_broadcast:
	VBROADCASTSS X0, Y6

	MOVQ logits+0(FP), SI
	MOVQ out+8(FP), DI
	MOVQ count+24(FP), CX

	MOVQ $samAvx2ExpC<>(SB), AX
	VMOVSS (AX), X8
	VBROADCASTSS X8, Y8
	VMOVSS 4(AX), X9
	VBROADCASTSS X9, Y9
	VMOVSS 12(AX), X11
	VBROADCASTSS X11, Y11
	VMOVSS 16(AX), X12
	VBROADCASTSS X12, Y12
	VMOVSS 20(AX), X13
	VBROADCASTSS X13, Y13
	VMOVSS 24(AX), X14
	VBROADCASTSS X14, Y14
	VMOVSS 28(AX), X15
	VBROADCASTSS X15, Y15
	VMOVSS 32(AX), X16
	VBROADCASTSS X16, Y16
	VMOVSS 36(AX), X17
	VBROADCASTSS X17, Y17
	MOVQ AX, R15
	MOVD samAvx2ExpBias127<>(SB), R14
	VMOVSS samAvx2SoftmaxClamp<>(SB), X8
	VBROADCASTSS X8, Y4
	VXORPS Y5, Y5, Y5

sam_avx2_softmax_exp_w8:
	CMPQ CX, $8
	JL   sam_avx2_softmax_exp_w4

	VMOVUPS (SI), Y0
	VSUBPS Y6, Y0, Y0
	VDIVPS Y10, Y0, Y0
	VMAXPS Y4, Y0, Y0
	VMULPS Y8, Y0, Y1
	VROUNDPS $8, Y1, Y1
	VMULPS Y1, Y9, Y2
	VSUBPS Y2, Y0, Y0
	VMOVAPS Y11, Y3
	VFMADD213PS Y3, Y0, Y11
	VMOVAPS Y12, Y3
	VFMADD213PS Y3, Y0, Y12
	VMOVAPS Y13, Y3
	VFMADD213PS Y3, Y0, Y13
	VMOVAPS Y14, Y3
	VFMADD213PS Y3, Y0, Y14
	VMOVAPS Y15, Y3
	VFMADD213PS Y3, Y0, Y15
	VMOVAPS Y16, Y3
	VFMADD213PS Y3, Y0, Y16
	VMOVAPS Y17, Y7
	VFMADD213PS Y7, Y0, Y17
	VCVTPS2DQ Y1, Y1
	MOVD R14, X3
	VPBROADCASTD X3, Y3
	VPADDD Y3, Y1, Y1
	VPSLLD $23, Y1, Y1
	VPADDD Y1, Y7, Y7
	VADDPS Y5, Y7, Y5
	VMOVUPS Y7, (DI)

	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $8, CX
	JMP  sam_avx2_softmax_exp_w8

sam_avx2_softmax_exp_w4:
	CMPQ CX, $4
	JL   sam_avx2_softmax_exp_tail

	VMOVUPS (SI), X0
	VSUBPS X6, X0, X0
	VDIVPS X10, X0, X0
	VMAXPS X4, X0, X0
	VMULPS X8, X0, X1
	VROUNDPS $8, X1, X1
	VMULPS X1, X9, X2
	VSUBPS X2, X0, X0
	VMOVAPS X11, X3
	VFMADD213PS X3, X0, X11
	VMOVAPS X12, X3
	VFMADD213PS X3, X0, X12
	VMOVAPS X13, X3
	VFMADD213PS X3, X0, X13
	VMOVAPS X14, X3
	VFMADD213PS X3, X0, X14
	VMOVAPS X15, X3
	VFMADD213PS X3, X0, X15
	VMOVAPS X16, X3
	VFMADD213PS X3, X0, X16
	VMOVAPS X17, X7
	VFMADD213PS X7, X0, X17
	VCVTPS2DQ X1, X1
	MOVD R14, X3
	VPBROADCASTD X3, X3
	VPADDD X3, X1, X1
	VPSLLD $23, X1, X1
	VPADDD X1, X7, X7
	VADDPS X5, X7, X5
	VMOVUPS X7, (DI)

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  sam_avx2_softmax_exp_w4

sam_avx2_softmax_exp_tail:
	TESTQ CX, CX
	JZ   sam_avx2_softmax_exp_reduce

	VMOVSS 32(R15), X16
	VBROADCASTSS X16, Y16
	VMOVSS 36(R15), X17
	VBROADCASTSS X17, Y17

sam_avx2_softmax_exp_scalar:
	MOVSS (SI), X0
	SUBSS X6, X0
	DIVSS X10, X0
	MAXSS X4, X0
	MOVSS X0, X1
	MULSS X8, X1
	ROUNDSS $8, X1, X1
	MOVSS X1, X2
	MULSS X9, X2
	SUBSS X2, X0
	VMOVSS 12(R15), X11
	VMOVSS 16(R15), X12
	VMOVSS 20(R15), X13
	VMOVSS 24(R15), X14
	VMOVSS 28(R15), X15
	MOVAPS X11, X3
	VFMADD213SS X3, X0, X11
	MOVAPS X12, X3
	VFMADD213SS X3, X0, X12
	MOVAPS X13, X3
	VFMADD213SS X3, X0, X13
	MOVAPS X14, X3
	VFMADD213SS X3, X0, X14
	MOVAPS X15, X3
	VFMADD213SS X3, X0, X15
	MOVSS 32(R15), X3
	MOVSS X3, X14
	VFMADD213SS X3, X0, X14
	MOVSS 36(R15), X7
	VFMADD213SS X7, X0, X7
	XORPS X2, X2
	MOVSS X1, X2
	VCVTPS2DQ X2, X2
	MOVD R14, X3
	PSHUFD $0, X3, X3
	PADDD X3, X2
	VPSLLD $23, X2, X2
	PADDD X2, X7
	ADDSS X7, X5
	MOVSS X7, (DI)
	ADDQ  $4, SI
	ADDQ  $4, DI
	DECQ  CX
	JNZ  sam_avx2_softmax_exp_scalar

sam_avx2_softmax_exp_reduce:
	VHADDPS Y5, Y5, Y5
	VHADDPS Y5, Y5, Y5
	VEXTRACTF128 $0, Y5, X0
	XORPS X1, X1
	UCOMISS X0, X1
	JE    sam_avx2_softmax_done

	MOVSS samAvx2OneF32<>(SB), X8
	DIVSS X0, X8
	VBROADCASTSS X8, Y8

	MOVQ out+8(FP), DI
	MOVQ count+24(FP), CX

sam_avx2_softmax_scale_w8:
	CMPQ CX, $8
	JL   sam_avx2_softmax_scale_w4

	VMOVUPS (DI), Y0
	VMULPS Y8, Y0, Y0
	VMOVUPS Y0, (DI)

	ADDQ $32, DI
	SUBQ $8, CX
	JMP  sam_avx2_softmax_scale_w8

sam_avx2_softmax_scale_w4:
	CMPQ CX, $4
	JL   sam_avx2_softmax_scale_tail

	VMOVUPS (DI), X0
	VMULPS X8, X0, X0
	VMOVUPS X0, (DI)

	ADDQ $16, DI
	SUBQ $4, CX
	JMP  sam_avx2_softmax_scale_w4

sam_avx2_softmax_scale_tail:
	TESTQ CX, CX
	JZ   sam_avx2_softmax_done

sam_avx2_softmax_scale_scalar:
	MOVSS (DI), X0
	MULSS X8, X0
	MOVSS X0, (DI)
	ADDQ  $4, DI
	DECQ  CX
	JNZ  sam_avx2_softmax_scale_scalar

sam_avx2_softmax_done:
	RET
