#include "textflag.h"

// func ActivationSteerFloat32AVX2Asm(dst, base, direction *float32, coefficient float32, count int)
TEXT ·ActivationSteerFloat32AVX2Asm(SB), NOSPLIT, $0-36
	MOVQ dst+0(FP), DI
	MOVQ base+8(FP), SI
	MOVQ direction+16(FP), R8
	MOVSS coefficient+24(FP), X15
	VBROADCASTSS X15, Y15
	MOVQ count+32(FP), CX

intrp_avx2_w8:
	CMPQ CX, $8
	JL   intrp_avx2_w4

	VMOVUPS (SI), Y0
	VMOVUPS (R8), Y1
	VFMADD231PS Y15, Y1, Y0
	VMOVUPS Y0, (DI)

	ADDQ $32, SI
	ADDQ $32, R8
	ADDQ $32, DI
	SUBQ $8, CX
	JMP  intrp_avx2_w8

intrp_avx2_w4:
	CMPQ CX, $4
	JL   intrp_avx2_tail

	VMOVUPS (SI), X0
	VMOVUPS (R8), X1
	VFMADD231PS X15, X1, X0
	VMOVUPS X0, (DI)

	ADDQ $16, SI
	ADDQ $16, R8
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  intrp_avx2_w4

intrp_avx2_tail:
	TESTQ CX, CX
	JZ   intrp_avx2_done

intrp_avx2_scalar:
	VMOVSS (SI), X0
	VMOVSS (R8), X1
	VFMADD231SS X15, X1, X0
	MOVSS X0, (DI)
	ADDQ $4, SI
	ADDQ $4, R8
	ADDQ $4, DI
	DECQ CX
	JNZ  intrp_avx2_scalar

intrp_avx2_done:
	RET
