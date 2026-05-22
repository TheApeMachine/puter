#include "textflag.h"

// func VsaBindFloat32SSE2Asm(dst, left, right *float32, count int)
TEXT ·VsaBindFloat32SSE2Asm(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ left+8(FP), SI
	MOVQ right+16(FP), R8
	MOVQ count+24(FP), CX

vsa_bind_sse2_w4:
	CMPQ CX, $4
	JL   vsa_bind_sse2_tail

	VMOVUPS (SI), X0
	VMOVUPS (R8), X1
	VMULPS  X1, X0, X0
	VMOVUPS X0, (DI)

	ADDQ $16, SI
	ADDQ $16, R8
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  vsa_bind_sse2_w4

vsa_bind_sse2_tail:
	TESTQ CX, CX
	JZ   vsa_bind_sse2_done

vsa_bind_sse2_scalar:
	MOVSS (SI), X0
	MOVSS (R8), X1
	MULSS X1, X0
	MOVSS X0, (DI)
	ADDQ  $4, SI
	ADDQ  $4, R8
	ADDQ  $4, DI
	DECQ  CX
	JNZ  vsa_bind_sse2_scalar

vsa_bind_sse2_done:
	RET

// func VsaBundleFloat32SSE2Asm(dst, left, right *float32, count int)
TEXT ·VsaBundleFloat32SSE2Asm(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ left+8(FP), SI
	MOVQ right+16(FP), R8
	MOVQ count+24(FP), CX

vsa_bundle_sse2_w4:
	CMPQ CX, $4
	JL   vsa_bundle_sse2_tail

	VMOVUPS (SI), X0
	VMOVUPS (R8), X1
	VADDPS  X1, X0, X0
	VMOVUPS X0, (DI)

	ADDQ $16, SI
	ADDQ $16, R8
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  vsa_bundle_sse2_w4

vsa_bundle_sse2_tail:
	TESTQ CX, CX
	JZ   vsa_bundle_sse2_done

vsa_bundle_sse2_scalar:
	MOVSS (SI), X0
	MOVSS (R8), X1
	ADDSS X1, X0
	MOVSS X0, (DI)
	ADDQ  $4, SI
	ADDQ  $4, R8
	ADDQ  $4, DI
	DECQ  CX
	JNZ  vsa_bundle_sse2_scalar

vsa_bundle_sse2_done:
	RET

// func VsaPermuteCopyFloat32SSE2Asm(dst, src *float32, count int)
TEXT ·VsaPermuteCopyFloat32SSE2Asm(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ count+16(FP), CX

vsa_copy_sse2_w4:
	CMPQ CX, $4
	JL   vsa_copy_sse2_tail

	VMOVUPS (SI), X0
	VMOVUPS X0, (DI)

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  vsa_copy_sse2_w4

vsa_copy_sse2_tail:
	TESTQ CX, CX
	JZ   vsa_copy_sse2_done

vsa_copy_sse2_scalar:
	MOVSS (SI), X0
	MOVSS X0, (DI)
	ADDQ  $4, SI
	ADDQ  $4, DI
	DECQ  CX
	JNZ  vsa_copy_sse2_scalar

vsa_copy_sse2_done:
	RET

// func VsaSimilarityFloat32SSE2Asm(left, right *float32, count int) float32
TEXT ·VsaSimilarityFloat32SSE2Asm(SB), NOSPLIT, $0-28
	MOVQ left+0(FP), SI
	MOVQ right+8(FP), DI
	MOVQ count+16(FP), CX

	TESTQ CX, CX
	JZ   vsa_sim_sse2_zero

	XORPD X0, X0

vsa_sim_sse2_w4:
	CMPQ CX, $4
	JL   vsa_sim_sse2_tail

	VMOVUPS (SI), X1
	VMOVUPS (DI), X2
	VCVTPS2PD X1, X3
	VCVTPS2PD X2, X4
	MULSD   X4, X3
	ADDSD   X3, X0

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  vsa_sim_sse2_w4

vsa_sim_sse2_tail:
	TESTQ CX, CX
	JZ   vsa_sim_sse2_reduce

vsa_sim_sse2_scalar:
	MOVSS (SI), X1
	MOVSS (DI), X2
	CVTSS2SD X1, X1
	CVTSS2SD X2, X2
	MULSD X2, X1
	ADDSD X1, X0
	ADDQ  $4, SI
	ADDQ  $4, DI
	DECQ  CX
	JNZ  vsa_sim_sse2_scalar

vsa_sim_sse2_reduce:
	ADDSD X0, X0
	MOVSD X0, X1
	UNPCKHPD X0, X1
	ADDSD X1, X0
	CVTSD2SS X0, X0
	MOVSS X0, ret+24(FP)
	RET

vsa_sim_sse2_zero:
	XORPS X0, X0
	MOVSS X0, ret+24(FP)
	RET
