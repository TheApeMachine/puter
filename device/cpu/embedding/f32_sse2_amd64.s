#include "textflag.h"

// func CopyRowFloat32SSE2Asm(dst, src *float32, hidden int)
TEXT ·CopyRowFloat32SSE2Asm(SB), NOSPLIT, $0-20
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ hidden+16(FP), CX

emb_copy_sse2_w4:
	CMPQ CX, $4
	JL   emb_copy_sse2_tail

	VMOVUPS (SI), X0
	VMOVUPS X0, (DI)

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  emb_copy_sse2_w4

emb_copy_sse2_tail:
	TESTQ CX, CX
	JZ   emb_copy_sse2_done

emb_copy_sse2_scalar:
	MOVSS (SI), X0
	MOVSS X0, (DI)
	ADDQ  $4, SI
	ADDQ  $4, DI
	DECQ  CX
	JNZ  emb_copy_sse2_scalar

emb_copy_sse2_done:
	RET

// func AddRowFloat32SSE2Asm(dst, src *float32, hidden int)
TEXT ·AddRowFloat32SSE2Asm(SB), NOSPLIT, $0-20
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ hidden+16(FP), CX

emb_add_sse2_w4:
	CMPQ CX, $4
	JL   emb_add_sse2_tail

	VMOVUPS (SI), X0
	VMOVUPS (DI), X1
	VADDPS  X0, X1, X0
	VMOVUPS X0, (DI)

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  emb_add_sse2_w4

emb_add_sse2_tail:
	TESTQ CX, CX
	JZ   emb_add_sse2_done

emb_add_sse2_scalar:
	VMOVSS (SI), X0
	VMOVSS (DI), X1
	VADDSS X0, X1, X0
	MOVSS X0, (DI)
	ADDQ  $4, SI
	ADDQ  $4, DI
	DECQ  CX
	JNZ  emb_add_sse2_scalar

emb_add_sse2_done:
	RET
