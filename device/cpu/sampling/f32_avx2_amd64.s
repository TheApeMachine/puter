#include "textflag.h"

// func GreedySampleFloat32AVX2Asm(logits *float32, count int) int32
TEXT ·GreedySampleFloat32AVX2Asm(SB), NOSPLIT, $0-16
	MOVQ logits+0(FP), SI
	MOVQ SI, BX
	MOVQ count+8(FP), CX

	TESTQ CX, CX
	JZ   greedy_avx2_zero

	CMPQ CX, $1
	JE   greedy_avx2_one

	MOVSS (SI), X0
	SHUFPS $0x00, X0, X0
	ADDQ $4, SI
	DECQ CX

greedy_avx2_max_w8:
	CMPQ CX, $8
	JL   greedy_avx2_max_w4

	VMOVUPS (SI), X1
	MAXPS X1, X0
	VMOVUPS 16(SI), X1
	MAXPS X1, X0

	ADDQ $32, SI
	SUBQ $8, CX
	JMP  greedy_avx2_max_w8

greedy_avx2_max_w4:
	CMPQ CX, $4
	JL   greedy_avx2_max_tail

	VMOVUPS (SI), X1
	MAXPS X1, X0

	ADDQ $16, SI
	SUBQ $4, CX
	JMP  greedy_avx2_max_w4

greedy_avx2_max_tail:
	TESTQ CX, CX
	JZ   greedy_avx2_max_done

greedy_avx2_max_scalar:
	MOVSS (SI), X1
	MAXSS X1, X0
	ADDQ  $4, SI
	DECQ  CX
	JNZ  greedy_avx2_max_scalar

greedy_avx2_max_done:
	MOVAPS X0, X4
	SHUFPS $0x4E, X0, X4
	MAXPS  X4, X0
	MOVAPS X0, X4
	SHUFPS $0xB1, X0, X4
	MAXPS  X4, X0
	SHUFPS $0x00, X0, X0

	MOVQ BX, SI
	MOVQ count+8(FP), CX
	XORQ R8, R8

greedy_avx2_find_w8:
	CMPQ CX, $8
	JL   greedy_avx2_find_w4

	VBROADCASTSS X0, Y1
	VMOVUPS (SI), Y2
	VCMPPS $0, Y1, Y2, Y2
	VEXTRACTF128 $0, Y2, X2
	PMOVMSKB X2, AX
	TESTQ AX, AX
	JNZ  greedy_avx2_found_lo

	VEXTRACTF128 $1, Y2, X2
	PMOVMSKB X2, AX
	TESTQ AX, AX
	JNZ  greedy_avx2_found_hi8

	ADDQ $32, SI
	ADDQ $8, R8
	SUBQ $8, CX
	JMP  greedy_avx2_find_w8

greedy_avx2_found_hi8:
	BSFQ AX, DX
	ADDQ $8, DX
	ADDQ R8, DX
	MOVL DX, ret+16(FP)
	RET

greedy_avx2_found_lo:
	BSFQ AX, DX
	ADDQ R8, DX
	MOVL DX, ret+16(FP)
	RET

greedy_avx2_find_w4:
	CMPQ CX, $4
	JL   greedy_avx2_find_tail

	MOVAPS X0, X1
	VMOVUPS (SI), X2
	VCMPPS $0, X1, X2, X3
	PMOVMSKB X3, AX
	TESTQ AX, AX
	JNZ  greedy_avx2_found

	ADDQ $16, SI
	ADDQ $4, R8
	SUBQ $4, CX
	JMP  greedy_avx2_find_w4

greedy_avx2_find_tail:
	TESTQ CX, CX
	JZ   greedy_avx2_fail

greedy_avx2_find_scalar:
	MOVSS (SI), X1
	UCOMISS X0, X1
	JNE  greedy_avx2_find_next
	MOVL R8, ret+16(FP)
	RET

greedy_avx2_find_next:
	ADDQ $4, SI
	INCQ R8
	DECQ CX
	JNZ  greedy_avx2_find_scalar

greedy_avx2_fail:
	MOVQ count+8(FP), AX
	DECQ AX
	MOVL AX, ret+16(FP)
	RET

greedy_avx2_found:
	BSFQ AX, DX
	ADDQ R8, DX
	MOVL DX, ret+16(FP)
	RET

greedy_avx2_one:
	XORL AX, AX
	MOVL AX, ret+16(FP)
	RET

greedy_avx2_zero:
	XORL AX, AX
	MOVL AX, ret+16(FP)
	RET
