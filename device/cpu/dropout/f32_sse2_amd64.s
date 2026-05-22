#include "textflag.h"

DATA dropSignFlipSSE2<>+0(SB)/4, $0x80000000
GLOBL dropSignFlipSSE2<>(SB), RODATA|NOPTR, $4

// func DropoutFloat32SSE2Asm(
//     dst, src *float32, count int,
//     seedLane *uint32, scale, threshold float32,
// )
TEXT ·DropoutFloat32SSE2Asm(SB), NOSPLIT, $0-40
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ count+16(FP), CX
	MOVQ seedLane+24(FP), R8
	MOVSS scale+32(FP), X15
	MOVSS threshold+36(FP), X14

	SHUFPS $0, X15, X15
	SHUFPS $0, X14, X14
	MOVSS dropSignFlipSSE2<>(SB), X13
	SHUFPS $0, X13, X13

	XORPS X11, X11
	MOVL  (R8), AX
	MOVD  AX, X11

	TESTQ CX, CX
	JZ   drop_sse2_done

drop_sse2_w4:
	CMPQ CX, $4
	JL   drop_sse2_tail

	MOVQ $4, R9
	LEAQ 0(SP), R10
drop_sse2_gen4:
	VPSLLD $13, X11, X12
	VPXOR X12, X11, X11
	VPSRLD $17, X11, X12
	VPXOR X12, X11, X11
	VPSLLD $5, X11, X12
	VPXOR X12, X11, X11
	VMOVD X11, (R10)
	ADDQ $4, R10
	DECQ R9
	JNZ  drop_sse2_gen4

	MOVUPS (SI), X0
	VMOVDQU (SP), X2
	VPXOR X13, X2, X2
	MOVAPS X14, X1
	VPXOR X13, X1, X1
	VPCMPGTD X1, X2, X12
	VANDPS X12, X0, X0
	VMULPS X15, X0, X0
	VMOVUPS X0, (DI)

	ADDQ $16, DI
	ADDQ $16, SI
	SUBQ $4, CX
	JMP  drop_sse2_w4

drop_sse2_tail:
	TESTQ CX, CX
	JZ   drop_sse2_done

	MOVQ CX, R9
drop_sse2_scalar:
	VPSLLD $13, X11, X12
	VPXOR X12, X11, X11
	VPSRLD $17, X11, X12
	VPXOR X12, X11, X11
	VPSLLD $5, X11, X12
	VPXOR X12, X11, X11
	VMOVD X11, AX
	MOVD X14, DX
	CMPL AX, DX
	JA   drop_sse2_keep_lane
	XORPS X0, X0
	JMP  drop_sse2_store_lane
drop_sse2_keep_lane:
	MOVSS (SI), X0
	MULSS X15, X0
drop_sse2_store_lane:
	MOVSS X0, (DI)
	ADDQ $4, DI
	ADDQ $4, SI
	DECQ R9
	JNZ  drop_sse2_scalar

drop_sse2_done:
	VMOVD X11, AX
	MOVL AX, (R8)
	RET
