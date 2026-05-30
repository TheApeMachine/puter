#include "textflag.h"

DATA geomCouplingBF16AbsMaskSSE2<>+0(SB)/4, $0x7fffffff
GLOBL geomCouplingBF16AbsMaskSSE2<>(SB), RODATA|NOPTR, $4

#define BF16_RNE_EAX \
	MOVL  AX, DX; \
	SHRL  $16, DX; \
	ANDL  $1, DX; \
	ADDL  $0x7fff, AX; \
	ADDL  DX, AX; \
	SHRL  $16, AX

// func PhaseCouplingBFloat16SSE2Asm(dst, left, right *uint16, count int)
TEXT ·PhaseCouplingBFloat16SSE2Asm(SB), NOSPLIT, $0-32
	MOVQ destination+0(FP), DI
	MOVQ leftGrowth+8(FP), SI
	MOVQ rightGrowth+16(FP), R8
	MOVQ count+24(FP), CX

	VBROADCASTSS geomCouplingBF16AbsMaskSSE2<>(SB), X7

pcbf16_sse2_loop:
	TESTQ CX, CX
	JZ    pcbf16_sse2_done

	MOVWLZX (SI), AX
	SHLQ  $16, AX
	VMOVD X2, AX
	MOVWLZX (R8), DX
	SHLQ  $16, DX
	VMOVD X3, DX
	VANDPS X7, X2, X4
	VANDPS X7, X3, X6
	VMULSS X6, X4, X4
	VSQRTSS X4, X4, X5
	MOVL  X5, AX
	BF16_RNE_EAX
	CMPL  AX, $0x3c23
	JAE   pcbf16_sse2_compute
	XORL  AX, AX
	JMP   pcbf16_sse2_store

pcbf16_sse2_compute:
	VMULSS X3, X2, X7
	VMULSS X5, X5, X5
	VDIVSS X5, X7, X7
	MOVL  X7, AX

pcbf16_sse2_store:
	BF16_RNE_EAX
	MOVW  AX, (DI)
	ADDQ  $2, SI
	ADDQ  $2, R8
	ADDQ  $2, DI
	DECQ  CX
	JMP   pcbf16_sse2_loop

pcbf16_sse2_done:
	VZEROUPPER
	RET
