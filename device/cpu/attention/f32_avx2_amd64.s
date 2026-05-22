#include "textflag.h"

// func FlashAttentionOnlineUpdateAVX2Asm(
//     acc, valueRow *float32,
//     alpha, shifted float32,
//     n int,
// )
//
// acc[i] = acc[i]*alpha + valueRow[i]*shifted for i in [0,n).
TEXT ·FlashAttentionOnlineUpdateAVX2Asm(SB), NOSPLIT, $0-32
	MOVQ acc+0(FP), SI
	MOVQ valueRow+8(FP), DI
	MOVQ n+24(FP), CX

	MOVSS alpha+16(FP), X14
	VBROADCASTSS X14, Y14
	MOVSS shifted+20(FP), X15
	VBROADCASTSS X15, Y15

flash_upd_avx2_w8:
	CMPQ CX, $8
	JL   flash_upd_avx2_w4

	VMOVUPS (SI), Y0
	VMOVUPS (DI), Y2
	VMULPS  Y14, Y0, Y0
	VMULPS  Y15, Y2, Y2
	VADDPS  Y2, Y0, Y0
	VMOVUPS Y0, (SI)

	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $8, CX
	JMP  flash_upd_avx2_w8

flash_upd_avx2_w4:
	CMPQ CX, $4
	JL   flash_upd_avx2_tail

	VMOVUPS (SI), X0
	VMOVUPS (DI), X2
	VMULPS  X14, X0, X0
	VMULPS  X15, X2, X2
	VADDPS  X2, X0, X0
	VMOVUPS X0, (SI)

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  flash_upd_avx2_w4

flash_upd_avx2_tail:
	TESTQ CX, CX
	JZ   flash_upd_avx2_done

flash_upd_avx2_scalar:
	MOVSS (SI), X0
	MOVSS (DI), X2
	VMULSS X14, X0, X0
	VMULSS X15, X2, X2
	VADDSS X2, X0, X0
	MOVSS X0, (SI)
	ADDQ  $4, SI
	ADDQ  $4, DI
	DECQ  CX
	JNZ  flash_upd_avx2_scalar

flash_upd_avx2_done:
	RET

// func FlashAttentionScaleAVX2Asm(
//     out, acc *float32,
//     invNormalizer float32,
//     n int,
// )
//
// out[i] = acc[i] * invNormalizer for i in [0,n).
TEXT ·FlashAttentionScaleAVX2Asm(SB), NOSPLIT, $0-32
	MOVQ out+0(FP), DI
	MOVQ acc+8(FP), SI
	MOVQ n+24(FP), CX

	MOVSS invNormalizer+16(FP), X15
	VBROADCASTSS X15, Y15

flash_scale_avx2_w8:
	CMPQ CX, $8
	JL   flash_scale_avx2_w4

	VMOVUPS (SI), Y0
	VMULPS  Y15, Y0, Y0
	VMOVUPS Y0, (DI)

	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $8, CX
	JMP  flash_scale_avx2_w8

flash_scale_avx2_w4:
	CMPQ CX, $4
	JL   flash_scale_avx2_tail

	VMOVUPS (SI), X0
	VMULPS  X15, X0, X0
	VMOVUPS X0, (DI)

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  flash_scale_avx2_w4

flash_scale_avx2_tail:
	TESTQ CX, CX
	JZ   flash_scale_avx2_done

flash_scale_avx2_scalar:
	MOVSS (SI), X0
	VMULSS X15, X0, X0
	MOVSS X0, (DI)
	ADDQ  $4, SI
	ADDQ  $4, DI
	DECQ  CX
	JNZ  flash_scale_avx2_scalar

flash_scale_avx2_done:
	RET
