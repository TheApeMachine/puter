#include "textflag.h"

DATA losAbsMaskAVX2<>+0(SB)/4, $0x7fffffff
DATA losAbsMaskAVX2<>+4(SB)/4, $0x7fffffff
DATA losAbsMaskAVX2<>+8(SB)/4, $0x7fffffff
DATA losAbsMaskAVX2<>+12(SB)/4, $0x7fffffff
GLOBL losAbsMaskAVX2<>(SB), RODATA|NOPTR, $16

// func MseSumFloat32AVX2Asm(predictions, targets *float32, count int) float32
TEXT ·MseSumFloat32AVX2Asm(SB), NOSPLIT, $0-28
	MOVQ predictions+0(FP), SI
	MOVQ targets+8(FP), DI
	MOVQ count+16(FP), CX

	TESTQ CX, CX
	JZ   mse_avx2_zero

	VXORPD Y0, Y0, Y0

mse_avx2_w8:
	CMPQ CX, $8
	JL   mse_avx2_w4

	VMOVUPS Y1, (SI)
	VMOVUPS Y2, (DI)
	VSUBPS  Y2, Y1, Y3
	VEXTRACTF128 $0, Y3, X4
	VCVTPS2PD X4, Y5
	VMULPD  Y5, Y5, Y5
	VADDPD  Y0, Y5, Y0
	VEXTRACTF128 $1, Y3, X4
	VCVTPS2PD X4, Y5
	VMULPD  Y5, Y5, Y5
	VADDPD  Y0, Y5, Y0

	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $8, CX
	JMP  mse_avx2_w8

mse_avx2_w4:
	CMPQ CX, $4
	JL   mse_avx2_tail

	VMOVUPS X1, (SI)
	VMOVUPS X2, (DI)
	VSUBPS  X2, X1, X3
	VCVTPS2PD X3, Y5
	VMULPD  Y5, Y5, Y5
	VADDPD  Y0, Y5, Y0

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  mse_avx2_w4

mse_avx2_tail:
	TESTQ CX, CX
	JZ   mse_avx2_reduce

mse_avx2_scalar:
	MOVSS (SI), X1
	MOVSS (DI), X2
	VSUBSS X2, X1, X3
	CVTSS2SD X3, X3
	MULSD  X3, X3
	ADDSD  X3, X0
	ADDQ  $4, SI
	ADDQ  $4, DI
	DECQ  CX
	JNZ  mse_avx2_scalar

mse_avx2_reduce:
	VHADDPD Y1, Y0, Y0
	VHADDPD Y1, Y1, Y1
	VEXTRACTF128 $0, Y1, X0
	CVTSD2SS X0, X0
	MOVSS X0, ret+24(FP)
	RET

mse_avx2_zero:
	XORPS X0, X0
	MOVSS X0, ret+24(FP)
	RET

// func MaeSumFloat32AVX2Asm(predictions, targets *float32, count int) float32
TEXT ·MaeSumFloat32AVX2Asm(SB), NOSPLIT, $0-28
	MOVQ predictions+0(FP), SI
	MOVQ targets+8(FP), DI
	MOVQ count+16(FP), CX

	TESTQ CX, CX
	JZ   mae_avx2_zero

	VXORPD Y0, Y0, Y0
	VBROADCASTSS losAbsMaskAVX2<>(SB), Y7
	VBROADCASTSS losAbsMaskAVX2<>(SB), X7

mae_avx2_w8:
	CMPQ CX, $8
	JL   mae_avx2_w4

	VMOVUPS Y1, (SI)
	VMOVUPS Y2, (DI)
	VSUBPS  Y2, Y1, Y3
	VANDPS  Y7, Y3, Y3
	VEXTRACTF128 $0, Y3, X4
	VCVTPS2PD X4, Y5
	VADDPD  Y0, Y5, Y0
	VEXTRACTF128 $1, Y3, X4
	VCVTPS2PD X4, Y5
	VADDPD  Y0, Y5, Y0

	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $8, CX
	JMP  mae_avx2_w8

mae_avx2_w4:
	CMPQ CX, $4
	JL   mae_avx2_tail

	VMOVUPS X1, (SI)
	VMOVUPS X2, (DI)
	VSUBPS  X2, X1, X3
	VANDPS  X7, X3, X3
	VCVTPS2PD X3, Y5
	VADDPD  Y0, Y5, Y0

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  mae_avx2_w4

mae_avx2_tail:
	TESTQ CX, CX
	JZ   mae_avx2_reduce

mae_avx2_scalar:
	MOVSS (SI), X1
	MOVSS (DI), X2
	VSUBSS X2, X1, X3
	ANDPS X7, X3
	CVTSS2SD X3, X3
	ADDSD  X3, X0
	ADDQ  $4, SI
	ADDQ  $4, DI
	DECQ  CX
	JNZ  mae_avx2_scalar

mae_avx2_reduce:
	VHADDPD Y1, Y0, Y0
	VHADDPD Y1, Y1, Y1
	VEXTRACTF128 $0, Y1, X0
	CVTSD2SS X0, X0
	MOVSS X0, ret+24(FP)
	RET

mae_avx2_zero:
	XORPS X0, X0
	MOVSS X0, ret+24(FP)
	RET
