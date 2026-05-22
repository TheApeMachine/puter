#include "textflag.h"

DATA maskZeroSSE2<>+0(SB)/4, $0.0
GLOBL maskZeroSSE2<>(SB), RODATA|NOPTR, $4

DATA maskNegInfSSE2<>+0(SB)/4, $0xFF800000
GLOBL maskNegInfSSE2<>(SB), RODATA|NOPTR, $4

// func ApplyMaskFloat32SSE2Asm(input, mask, output *float32, count int)
TEXT ·ApplyMaskFloat32SSE2Asm(SB), NOSPLIT, $0-32
	MOVQ input+0(FP), DI
	MOVQ mask+8(FP), SI
	MOVQ output+16(FP), R8
	MOVQ count+24(FP), CX

mask_sse2_w4:
	CMPQ CX, $4
	JL   mask_sse2_tail

	VMOVUPS (DI), X0
	VMOVUPS (SI), X1
	VADDPS X1, X0, X0
	VMOVUPS X0, (R8)

	ADDQ $16, DI
	ADDQ $16, SI
	ADDQ $16, R8
	SUBQ $4, CX
	JMP  mask_sse2_w4

mask_sse2_tail:
	TESTQ CX, CX
	JZ   mask_sse2_done

mask_sse2_scalar:
	VMOVSS (DI), X0
	VMOVSS (SI), X1
	VADDSS X1, X0, X0
	MOVSS X0, (R8)
	ADDQ $4, DI
	ADDQ $4, SI
	ADDQ $4, R8
	DECQ CX
	JNZ  mask_sse2_scalar

mask_sse2_done:
	RET

// func CausalMaskFloat32SSE2Asm(output *float32, seqQ, seqK int)
TEXT ·CausalMaskFloat32SSE2Asm(SB), NOSPLIT, $0-24
	MOVQ output+0(FP), DI
	MOVQ seqQ+8(FP), R10
	MOVQ seqK+16(FP), BX

	MOVSS maskZeroSSE2<>(SB), X2
	MOVSS maskNegInfSSE2<>(SB), X3

	XORQ R11, R11

causal_sse2_row:
	CMPQ R11, R10
	JGE  causal_sse2_done

	MOVQ R11, AX
	INCQ AX
	CMPQ AX, BX
	JLE  causal_sse2_zero_len_ok
	MOVQ BX, AX

causal_sse2_zero_len_ok:
	MOVQ AX, CX

causal_sse2_zero_w4:
	CMPQ CX, $4
	JL   causal_sse2_zero_tail

	MOVUPS X2, (DI)

	ADDQ $16, DI
	SUBQ $4, CX
	JMP  causal_sse2_zero_w4

causal_sse2_zero_tail:
	TESTQ CX, CX
	JZ   causal_sse2_zero_done

causal_sse2_zero_scalar:
	MOVSS X2, (DI)
	ADDQ $4, DI
	DECQ CX
	JNZ  causal_sse2_zero_scalar

causal_sse2_zero_done:
	MOVQ seqK+16(FP), BX
	MOVQ R11, AX
	INCQ AX
	CMPQ AX, BX
	JGE  causal_sse2_next_row

	MOVQ BX, R12
	SUBQ AX, R12
	MOVQ R12, CX

causal_sse2_inf_w4:
	CMPQ CX, $4
	JL   causal_sse2_inf_tail

	MOVUPS X3, (DI)

	ADDQ $16, DI
	SUBQ $4, CX
	JMP  causal_sse2_inf_w4

causal_sse2_inf_tail:
	TESTQ CX, CX
	JZ   causal_sse2_next_row

causal_sse2_inf_scalar:
	MOVSS X3, (DI)
	ADDQ $4, DI
	DECQ CX
	JNZ  causal_sse2_inf_scalar

causal_sse2_next_row:
	INCQ R11
	JMP  causal_sse2_row

causal_sse2_done:
	RET

DATA maskIota4SSE2<>+0(SB)/4, $0
DATA maskIota4SSE2<>+4(SB)/4, $1
DATA maskIota4SSE2<>+8(SB)/4, $2
DATA maskIota4SSE2<>+12(SB)/4, $3
GLOBL maskIota4SSE2<>(SB), RODATA|NOPTR, $16

// func ALiBiBiasFloat32SSE2Asm(scores, slope, output *float32, seqQ, seqK int)
TEXT ·ALiBiBiasFloat32SSE2Asm(SB), NOSPLIT, $0-40
	MOVQ scores+0(FP), SI
	MOVQ slope+8(FP), R9
	MOVQ output+16(FP), DI
	MOVQ seqQ+24(FP), R10
	MOVQ seqK+32(FP), BX

	MOVSS (R9), X15

	XORQ R11, R11

alibi_sse2_row:
	CMPQ R11, R10
	JGE  alibi_sse2_done

	VMOVD R11, X13

	XORQ R12, R12

alibi_sse2_col:
	MOVQ BX, CX
	SUBQ R12, CX
	JZ   alibi_sse2_row_done

	CMPQ CX, $4
	JL   alibi_sse2_col_tail

	VMOVUPS (SI), X0
	VMOVD R12, X10
	VPBROADCASTD X10, X11
	VPADDD maskIota4SSE2<>(SB), X11, X11
	VPSUBD X11, X13, X12
	VCVTDQ2PS X12, X10
	VXORPS X9, X9, X9
	CMPPS $1, X10, X9, X8
	VMULPS X15, X10, X11
	VSUBPS X11, X0, X1
	VANDPS X8, X1, X4
	VANDNPS X8, X0, X5
	VORPS X4, X5, X0
	VMOVUPS X0, (DI)

	ADDQ $16, SI
	ADDQ $16, DI
	ADDQ $4, R12
	JMP  alibi_sse2_col

alibi_sse2_col_tail:
	TESTQ CX, CX
	JZ   alibi_sse2_row_done

alibi_sse2_col_scalar:
	VMOVSS (SI), X0
	MOVQ R11, AX
	MOVQ R12, DX
	SUBQ DX, AX
	CMPQ AX, $0
	JL   alibi_sse2_keep_score

	XORPS X9, X9
	CVTSI2SS AX, X1
	VMULSS X15, X1, X1
	VSUBSS X1, X0, X0

alibi_sse2_keep_score:
	MOVSS X0, (DI)
	ADDQ $4, SI
	ADDQ $4, DI
	INCQ R12
	DECQ CX
	JNZ  alibi_sse2_col_scalar

alibi_sse2_row_done:
	INCQ R11
	JMP  alibi_sse2_row

alibi_sse2_done:
	RET
