#include "textflag.h"

DATA losAbsMaskSSE2<>+0(SB)/4, $0x7fffffff
DATA losAbsMaskSSE2<>+4(SB)/4, $0x7fffffff
DATA losAbsMaskSSE2<>+8(SB)/4, $0x7fffffff
DATA losAbsMaskSSE2<>+12(SB)/4, $0x7fffffff
GLOBL losAbsMaskSSE2<>(SB), RODATA|NOPTR, $16

// func MseSumFloat32SSE2Asm(predictions, targets *float32, count int) float32
TEXT ·MseSumFloat32SSE2Asm(SB), NOSPLIT, $0-28
	MOVQ predictions+0(FP), SI
	MOVQ targets+8(FP), DI
	MOVQ count+16(FP), CX

	TESTQ CX, CX
	JZ   mse_sse2_zero

	XORPD X0, X0

mse_sse2_w4:
	CMPQ CX, $4
	JL   mse_sse2_tail

	MOVUPS X1, (SI)
	MOVUPS X2, (DI)
	SUBPS  X2, X1
	CVTPS2PD X1, X3
	MULPD  X3, X3
	ADDPD  X3, X0

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  mse_sse2_w4

mse_sse2_tail:
	TESTQ CX, CX
	JZ   mse_sse2_reduce

mse_sse2_scalar:
	MOVSS (SI), X1
	MOVSS (DI), X2
	SUBSS X2, X1
	CVTSS2SD X1, X1
	MULSD  X1, X1
	ADDSD  X1, X0
	ADDQ  $4, SI
	ADDQ  $4, DI
	DECQ  CX
	JNZ  mse_sse2_scalar

mse_sse2_reduce:
	ADDSD X0, X0
	MOVSD X0, X1
	UNPCKHPD X0, X1
	ADDSD X1, X0
	CVTSD2SS X0, X0
	MOVSS X0, ret+24(FP)
	RET

mse_sse2_zero:
	XORPS X0, X0
	MOVSS X0, ret+24(FP)
	RET

// func MaeSumFloat32SSE2Asm(predictions, targets *float32, count int) float32
TEXT ·MaeSumFloat32SSE2Asm(SB), NOSPLIT, $0-28
	MOVQ predictions+0(FP), SI
	MOVQ targets+8(FP), DI
	MOVQ count+16(FP), CX

	TESTQ CX, CX
	JZ   mae_sse2_zero

	XORPD X0, X0
	MOVUPS losAbsMaskSSE2<>(SB), X7

mae_sse2_w4:
	CMPQ CX, $4
	JL   mae_sse2_tail

	MOVUPS X1, (SI)
	MOVUPS X2, (DI)
	SUBPS  X2, X1
	ANDPS  X7, X1
	CVTPS2PD X1, X3
	ADDPD  X3, X0

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  mae_sse2_w4

mae_sse2_tail:
	TESTQ CX, CX
	JZ   mae_sse2_reduce

mae_sse2_scalar:
	MOVSS (SI), X1
	MOVSS (DI), X2
	SUBSS X2, X1
	ANDPS X7, X1
	CVTSS2SD X1, X1
	ADDSD  X1, X0
	ADDQ  $4, SI
	ADDQ  $4, DI
	DECQ  CX
	JNZ  mae_sse2_scalar

mae_sse2_reduce:
	ADDSD X0, X0
	MOVSD X0, X1
	UNPCKHPD X0, X1
	ADDSD X1, X0
	CVTSD2SS X0, X0
	MOVSS X0, ret+24(FP)
	RET

mae_sse2_zero:
	XORPS X0, X0
	MOVSS X0, ret+24(FP)
	RET
