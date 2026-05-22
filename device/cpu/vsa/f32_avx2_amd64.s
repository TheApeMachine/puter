#include "textflag.h"

// func VsaBindFloat32AVX2Asm(dst, left, right *float32, count int)
TEXT ·VsaBindFloat32AVX2Asm(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ left+8(FP), SI
	MOVQ right+16(FP), R8
	MOVQ count+24(FP), CX

vsa_bind_avx2_w8:
	CMPQ CX, $8
	JL   vsa_bind_avx2_w4

	VMOVUPS (SI), Y0
	VMOVUPS (R8), Y1
	VMULPS  Y1, Y0, Y0
	VMOVUPS Y0, (DI)

	ADDQ $32, SI
	ADDQ $32, R8
	ADDQ $32, DI
	SUBQ $8, CX
	JMP  vsa_bind_avx2_w8

vsa_bind_avx2_w4:
	CMPQ CX, $4
	JL   vsa_bind_avx2_tail

	VMOVUPS (SI), X0
	VMOVUPS (R8), X1
	VMULPS  X1, X0, X0
	VMOVUPS X0, (DI)

	ADDQ $16, SI
	ADDQ $16, R8
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  vsa_bind_avx2_w4

vsa_bind_avx2_tail:
	TESTQ CX, CX
	JZ   vsa_bind_avx2_done

vsa_bind_avx2_scalar:
	MOVSS (SI), X0
	MOVSS (R8), X1
	MULSS X1, X0
	MOVSS X0, (DI)
	ADDQ  $4, SI
	ADDQ  $4, R8
	ADDQ  $4, DI
	DECQ  CX
	JNZ  vsa_bind_avx2_scalar

vsa_bind_avx2_done:
	RET

// func VsaBundleFloat32AVX2Asm(dst, left, right *float32, count int)
TEXT ·VsaBundleFloat32AVX2Asm(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ left+8(FP), SI
	MOVQ right+16(FP), R8
	MOVQ count+24(FP), CX

vsa_bundle_avx2_w8:
	CMPQ CX, $8
	JL   vsa_bundle_avx2_w4

	VMOVUPS (SI), Y0
	VMOVUPS (R8), Y1
	VADDPS  Y1, Y0, Y0
	VMOVUPS Y0, (DI)

	ADDQ $32, SI
	ADDQ $32, R8
	ADDQ $32, DI
	SUBQ $8, CX
	JMP  vsa_bundle_avx2_w8

vsa_bundle_avx2_w4:
	CMPQ CX, $4
	JL   vsa_bundle_avx2_tail

	VMOVUPS (SI), X0
	VMOVUPS (R8), X1
	VADDPS  X1, X0, X0
	VMOVUPS X0, (DI)

	ADDQ $16, SI
	ADDQ $16, R8
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  vsa_bundle_avx2_w4

vsa_bundle_avx2_tail:
	TESTQ CX, CX
	JZ   vsa_bundle_avx2_done

vsa_bundle_avx2_scalar:
	MOVSS (SI), X0
	MOVSS (R8), X1
	ADDSS X1, X0
	MOVSS X0, (DI)
	ADDQ  $4, SI
	ADDQ  $4, R8
	ADDQ  $4, DI
	DECQ  CX
	JNZ  vsa_bundle_avx2_scalar

vsa_bundle_avx2_done:
	RET

// func VsaPermuteCopyFloat32AVX2Asm(dst, src *float32, count int)
TEXT ·VsaPermuteCopyFloat32AVX2Asm(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ count+16(FP), CX

vsa_copy_avx2_w8:
	CMPQ CX, $8
	JL   vsa_copy_avx2_w4

	VMOVUPS (SI), Y0
	VMOVUPS Y0, (DI)

	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $8, CX
	JMP  vsa_copy_avx2_w8

vsa_copy_avx2_w4:
	CMPQ CX, $4
	JL   vsa_copy_avx2_tail

	VMOVUPS (SI), X0
	VMOVUPS X0, (DI)

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  vsa_copy_avx2_w4

vsa_copy_avx2_tail:
	TESTQ CX, CX
	JZ   vsa_copy_avx2_done

vsa_copy_avx2_scalar:
	MOVSS (SI), X0
	MOVSS X0, (DI)
	ADDQ  $4, SI
	ADDQ  $4, DI
	DECQ  CX
	JNZ  vsa_copy_avx2_scalar

vsa_copy_avx2_done:
	RET

// func VsaSimilarityFloat32AVX2Asm(left, right *float32, count int) float32
TEXT ·VsaSimilarityFloat32AVX2Asm(SB), NOSPLIT, $0-28
	MOVQ left+0(FP), SI
	MOVQ right+8(FP), DI
	MOVQ count+16(FP), CX

	TESTQ CX, CX
	JZ   vsa_sim_avx2_zero

	VXORPD Y0, Y0, Y0

vsa_sim_avx2_w8:
	CMPQ CX, $8
	JL   vsa_sim_avx2_w4

	VMOVUPS (SI), Y1
	VMOVUPS (DI), Y2
	VEXTRACTF128 $0, Y1, X3
	VEXTRACTF128 $0, Y2, X4
	VCVTPS2PD X3, Y5
	VCVTPS2PD X4, Y6
	VFMADD231PD Y0, Y6, Y5
	VEXTRACTF128 $1, Y1, X3
	VEXTRACTF128 $1, Y2, X4
	VCVTPS2PD X3, Y5
	VCVTPS2PD X4, Y6
	VFMADD231PD Y0, Y6, Y5

	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $8, CX
	JMP  vsa_sim_avx2_w8

vsa_sim_avx2_w4:
	CMPQ CX, $4
	JL   vsa_sim_avx2_tail

	VMOVUPS (SI), X1
	VMOVUPS (DI), X2
	VCVTPS2PD X1, Y5
	VCVTPS2PD X2, Y6
	VFMADD231PD Y0, Y6, Y5

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  vsa_sim_avx2_w4

vsa_sim_avx2_tail:
	TESTQ CX, CX
	JZ   vsa_sim_avx2_reduce

vsa_sim_avx2_scalar:
	MOVSS (SI), X1
	MOVSS (DI), X2
	CVTSS2SD X1, X1
	CVTSS2SD X2, X2
	MULSD X2, X1
	ADDSD X1, X0
	ADDQ  $4, SI
	ADDQ  $4, DI
	DECQ  CX
	JNZ  vsa_sim_avx2_scalar

vsa_sim_avx2_reduce:
	VHADDPD Y1, Y0, Y0
	VHADDPD Y1, Y1, Y1
	VEXTRACTF128 $0, Y1, X0
	CVTSD2SS X0, X0
	MOVSS X0, ret+24(FP)
	RET

vsa_sim_avx2_zero:
	XORPS X0, X0
	MOVSS X0, ret+24(FP)
	RET
