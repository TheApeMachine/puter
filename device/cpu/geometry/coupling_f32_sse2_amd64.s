#include "textflag.h"

DATA geomCouplingAbsMaskSSE2<>+0(SB)/4, $0x7fffffff
DATA geomCouplingAbsMaskSSE2<>+4(SB)/4, $0x7fffffff
DATA geomCouplingAbsMaskSSE2<>+8(SB)/4, $0x7fffffff
DATA geomCouplingAbsMaskSSE2<>+12(SB)/4, $0x7fffffff
GLOBL geomCouplingAbsMaskSSE2<>(SB), RODATA|NOPTR, $16

DATA geomCouplingEpsSSE2<>+0(SB)/4, $0x3c23d70a
GLOBL geomCouplingEpsSSE2<>(SB), RODATA|NOPTR, $4

DATA geomCouplingZeroSSE2<>+0(SB)/4, $0x00000000
GLOBL geomCouplingZeroSSE2<>(SB), RODATA|NOPTR, $4

// func PhaseCouplingFloat32SSE2Asm(dst, left, right *float32, count int)
TEXT ·PhaseCouplingFloat32SSE2Asm(SB), NOSPLIT, $0-32
	MOVQ destination+0(FP), DI
	MOVQ leftGrowth+8(FP), SI
	MOVQ rightGrowth+16(FP), R8
	MOVQ count+24(FP), CX

	MOVAPS geomCouplingAbsMaskSSE2<>(SB), X30
	MOVSS geomCouplingEpsSSE2<>(SB), X29
	MOVSS geomCouplingZeroSSE2<>(SB), X28

pc_sse2_w4:
	CMPQ CX, $4
	JL pc_sse2_tail

	MOVUPS (SI), X0
	MOVUPS (R8), X1
	ANDPS X30, X0
	ANDPS X30, X1
	MOVAPS X0, X2
	MOVAPS X1, X3
	MULPS X3, X2
	SQRTPS X2, X5
	CMPPS $1, X29, X5
	MOVAPS X0, X7
	MULPS X1, X7
	MOVAPS X5, X6
	MULPS X6, X6
	DIVPS X6, X7
	MOVAPS X28, X8
	MOVAPS X7, X9
	MOVAPS X5, X10
	ANDNPS X10, X9
	ANDPS X5, X8
	ORPS X8, X9
	MOVUPS X9, (DI)

	ADDQ $16, SI
	ADDQ $16, R8
	ADDQ $16, DI
	SUBQ $4, CX
	JMP pc_sse2_w4

pc_sse2_tail:
	TESTQ CX, CX
	JZ pc_sse2_done

pc_sse2_scalar:
	MOVSS (SI), X0
	MOVSS (R8), X1
	ANDPS X30, X0
	ANDPS X30, X1
	MOVSS X0, X2
	MULSS X1, X2
	SQRTSS X2, X5
	CMPSS $1, X29, X5
	MOVSS X0, X7
	MULSS X1, X7
	MOVSS X5, X6
	MULSS X6, X6
	DIVSS X6, X7
	MOVSS X28, X8
	MOVSS X7, X9
	MOVSS X5, X10
	ANDNPS X10, X9
	ANDPS X5, X8
	ORPS X8, X9
	MOVSS X9, (DI)
	ADDQ $4, SI
	ADDQ $4, R8
	ADDQ $4, DI
	DECQ CX
	JNZ pc_sse2_scalar

pc_sse2_done:
	RET
