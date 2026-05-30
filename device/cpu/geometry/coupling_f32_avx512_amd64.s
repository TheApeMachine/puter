#include "textflag.h"

DATA geomCouplingAbsMask<>+0(SB)/4, $0x7fffffff
DATA geomCouplingAbsMask<>+4(SB)/4, $0x7fffffff
DATA geomCouplingAbsMask<>+8(SB)/4, $0x7fffffff
DATA geomCouplingAbsMask<>+12(SB)/4, $0x7fffffff
GLOBL geomCouplingAbsMask<>(SB), RODATA|NOPTR, $16

DATA geomCouplingEps<>+0(SB)/4, $0x3c23d70a
GLOBL geomCouplingEps<>(SB), RODATA|NOPTR, $4

DATA geomCouplingZero<>+0(SB)/4, $0x00000000
GLOBL geomCouplingZero<>(SB), RODATA|NOPTR, $4

// func PhaseCouplingFloat32AVX512Asm(dst, left, right *float32, count int)
TEXT ·PhaseCouplingFloat32AVX512Asm(SB), NOSPLIT, $0-32
	MOVQ destination+0(FP), DI
	MOVQ leftGrowth+8(FP), SI
	MOVQ rightGrowth+16(FP), R8
	MOVQ count+24(FP), CX

	VBROADCASTSS geomCouplingAbsMask<>(SB), Z30
	VBROADCASTSS geomCouplingEps<>(SB), Z29
	VBROADCASTSS geomCouplingZero<>(SB), Z28

pc_avx512_w16:
	CMPQ CX, $16
	JL pc_avx512_w8

	VMOVUPS (SI), Z0
	VMOVUPS (R8), Z1
	VANDPS Z30, Z0, Z2
	VANDPS Z30, Z1, Z3
	VMULPS Z2, Z3, Z4
	VSQRTPS Z4, Z5
	VCMPPS $1, Z29, Z5, K1
	VMULPS Z0, Z1, Z6
	VMULPS Z5, Z5, Z7
	VDIVPS Z7, Z6, Z6
	VBLENDMPS Z28, Z6, K1, Z9
	VMOVUPS Z9, (DI)

	ADDQ $64, SI
	ADDQ $64, R8
	ADDQ $64, DI
	SUBQ $16, CX
	JMP pc_avx512_w16

pc_avx512_w8:
	CMPQ CX, $8
	JL pc_avx512_w4

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
	JMP pc_avx512_w8

pc_avx512_w4:
	CMPQ CX, $4
	JL pc_avx512_tail

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
	VBLENDVPS X28, X7, X6, X8
	VMOVUPS X8, (DI)

	ADDQ $16, SI
	ADDQ $16, R8
	ADDQ $16, DI
	SUBQ $4, CX
	JMP pc_avx512_w4

pc_avx512_tail:
	TESTQ CX, CX
	JZ pc_avx512_done

pc_avx512_scalar:
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
	VBLENDVPS X28, X7, X6, X8
	VMOVSS X8, (DI)
	ADDQ $4, SI
	ADDQ $4, R8
	ADDQ $4, DI
	DECQ CX
	JNZ pc_avx512_scalar

pc_avx512_done:
	VZEROUPPER
	RET
