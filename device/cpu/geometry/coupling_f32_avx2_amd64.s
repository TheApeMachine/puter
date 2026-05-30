#include "textflag.h"

DATA geomCouplingAbsMaskAVX2<>+0(SB)/4, $0x7fffffff
DATA geomCouplingAbsMaskAVX2<>+4(SB)/4, $0x7fffffff
DATA geomCouplingAbsMaskAVX2<>+8(SB)/4, $0x7fffffff
DATA geomCouplingAbsMaskAVX2<>+12(SB)/4, $0x7fffffff
GLOBL geomCouplingAbsMaskAVX2<>(SB), RODATA|NOPTR, $16

DATA geomCouplingEpsAVX2<>+0(SB)/4, $0x3c23d70a
GLOBL geomCouplingEpsAVX2<>(SB), RODATA|NOPTR, $4

DATA geomCouplingZeroAVX2<>+0(SB)/4, $0x00000000
GLOBL geomCouplingZeroAVX2<>(SB), RODATA|NOPTR, $4

// func PhaseCouplingFloat32AVX2Asm(dst, left, right *float32, count int)
TEXT ·PhaseCouplingFloat32AVX2Asm(SB), NOSPLIT, $0-32
	MOVQ destination+0(FP), DI
	MOVQ leftGrowth+8(FP), SI
	MOVQ rightGrowth+16(FP), R8
	MOVQ count+24(FP), CX

	VBROADCASTSS geomCouplingAbsMaskAVX2<>(SB), Y30
	VBROADCASTSS geomCouplingEpsAVX2<>(SB), Y29
	VBROADCASTSS geomCouplingZeroAVX2<>(SB), Y28

pc_avx2_w8:
	CMPQ CX, $8
	JL pc_avx2_w4

	VMOVUPS (SI), Y0
	VMOVUPS (R8), Y1
	VANDPS Y30, Y0, Y2
	VANDPS Y30, Y1, Y3
	VMULPS Y2, Y3, Y4
	VSQRTPS Y4, Y5
	VCMPPS $1, Y29, Y5, Y6
	VMULPS Y0, Y1, Y7
	VMULPS Y5, Y5, Y5
	VDIVPS Y5, Y7, Y7
	VBLENDVPS Y28, Y7, Y6, Y8
	VMOVUPS Y8, (DI)

	ADDQ $32, SI
	ADDQ $32, R8
	ADDQ $32, DI
	SUBQ $8, CX
	JMP pc_avx2_w8

pc_avx2_w4:
	CMPQ CX, $4
	JL pc_avx2_tail

	VMOVUPS (SI), X0
	VMOVUPS (R8), X1
	VANDPS X30, X0, X2
	VANDPS X30, X1, X3
	VMULPS X2, X3, X4
	VSQRTPS X4, X5
	VCMPPS $1, X29, X5, X6
	VMULPS X0, X1, X7
	VMULPS X5, X5, X5
	VDIVPS X5, X7, X7
	VBLENDVPS X7, X28, X6, X8
	VMOVUPS X8, (DI)

	ADDQ $16, SI
	ADDQ $16, R8
	ADDQ $16, DI
	SUBQ $4, CX
	JMP pc_avx2_w4

pc_avx2_tail:
	TESTQ CX, CX
	JZ pc_avx2_done

pc_avx2_scalar:
	VMOVSS (SI), X0
	VMOVSS (R8), X1
	VANDPS X30, X0, X2
	VANDPS X30, X1, X3
	VMULSS X2, X3, X4
	VSQRTSS X4, X5
	VCMPSS $1, X29, X5, X6
	VMULSS X0, X1, X7
	VMULSS X5, X5, X5
	VDIVSS X5, X7, X7
	VBLENDVPS X7, X28, X6, X8
	VMOVSS X8, (DI)
	ADDQ $4, SI
	ADDQ $4, R8
	ADDQ $4, DI
	DECQ CX
	JNZ pc_avx2_scalar

pc_avx2_done:
	VZEROUPPER
	RET
