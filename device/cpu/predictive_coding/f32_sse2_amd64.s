#include "textflag.h"

// func PCPredictionErrorFloat32SSE2Asm(observed, predicted, output *float32, count int)
TEXT ·PCPredictionErrorFloat32SSE2Asm(SB), NOSPLIT, $0-28
	MOVQ observed+0(FP), SI
	MOVQ predicted+8(FP), DI
	MOVQ output+16(FP), BX
	MOVQ count+24(FP), CX

pc_pe_sse2_w4:
	CMPQ CX, $4
	JL   pc_pe_sse2_tail

	VMOVUPS (SI), X0
	VMOVUPS (DI), X1
	SUBPS   X1, X0
	VMOVUPS X0, (BX)

	ADDQ $16, SI
	ADDQ $16, DI
	ADDQ $16, BX
	SUBQ $4, CX
	JMP  pc_pe_sse2_w4

pc_pe_sse2_tail:
	TESTQ CX, CX
	JZ   pc_pe_sse2_done

pc_pe_sse2_scalar:
	MOVSS (SI), X0
	MOVSS (DI), X1
	SUBSS X1, X0
	MOVSS X0, (BX)
	ADDQ  $4, SI
	ADDQ  $4, DI
	ADDQ  $4, BX
	DECQ  CX
	JNZ  pc_pe_sse2_scalar

pc_pe_sse2_done:
	RET

// func PCPredictionFloat32SSE2Asm(weights, representation, output *float32, outDim, inDim int)
TEXT ·PCPredictionFloat32SSE2Asm(SB), NOSPLIT, $0-36
	MOVQ weights+0(FP), R11
	MOVQ representation+8(FP), R12
	MOVQ output+16(FP), DI
	MOVQ outDim+24(FP), R9
	MOVQ inDim+32(FP), R8

pc_pred_sse2_row:
	TESTQ R9, R9
	JZ   pc_pred_sse2_done

	MOVQ R11, SI
	MOVQ R12, DX
	MOVQ R8, CX

	XORPD X0, X0

pc_pred_sse2_dot_w4:
	CMPQ CX, $4
	JL   pc_pred_sse2_dot_tail

	VMOVUPS (SI), X1
	VMOVUPS (DX), X2
	VCVTPS2PD X1, X3
	VCVTPS2PD X2, X4
	MULPD   X4, X3
	ADDPD   X3, X0

	MOVAPS X1, X5
	SHUFPS $0xEE, X1, X5
	MOVAPS X2, X6
	SHUFPS $0xEE, X2, X6
	VCVTPS2PD X5, X3
	VCVTPS2PD X6, X4
	MULPD   X4, X3
	ADDPD   X3, X0

	ADDQ $16, SI
	ADDQ $16, DX
	SUBQ $4, CX
	JMP  pc_pred_sse2_dot_w4

pc_pred_sse2_dot_tail:
	TESTQ CX, CX
	JZ   pc_pred_sse2_dot_reduce

pc_pred_sse2_dot_scalar:
	MOVSS (SI), X1
	MOVSS (DX), X2
	CVTSS2SD X1, X1
	CVTSS2SD X2, X2
	MULSD X2, X1
	ADDSD X1, X0
	ADDQ  $4, SI
	ADDQ  $4, DX
	DECQ  CX
	JNZ  pc_pred_sse2_dot_scalar

pc_pred_sse2_dot_reduce:
	MOVAPD X0, X1
	SHUFPD $1, X0, X1
	ADDPD  X1, X0
	CVTSD2SS X0, X0
	MOVSS X0, (DI)

	ADDQ $4, DI
	MOVQ R8, AX
	SHLQ $2, AX
	ADDQ AX, R11
	DECQ R9
	JMP  pc_pred_sse2_row

pc_pred_sse2_done:
	RET

// func PCUpdateRepresentationFloat32SSE2Asm(weights, representation, predictionError, output *float32, learningRate float32, outDim, inDim int)
TEXT ·PCUpdateRepresentationFloat32SSE2Asm(SB), NOSPLIT, $0-56
	MOVQ weights+0(FP), R11
	MOVQ representation+8(FP), SI
	MOVQ output+24(FP), DI
	MOVSS learningRate+32(FP), X15
	MOVQ outDim+40(FP), R9
	MOVQ inDim+48(FP), R8

	MOVQ DI, BX
	MOVQ R8, CX

pc_ur_sse2_copy_w4:
	CMPQ CX, $4
	JL   pc_ur_sse2_copy_tail

	VMOVUPS (SI), X0
	VMOVUPS X0, (BX)

	ADDQ $16, SI
	ADDQ $16, BX
	SUBQ $4, CX
	JMP  pc_ur_sse2_copy_w4

pc_ur_sse2_copy_tail:
	TESTQ CX, CX
	JZ   pc_ur_sse2_rows

pc_ur_sse2_copy_scalar:
	MOVSS (SI), X0
	MOVSS X0, (BX)
	ADDQ  $4, SI
	ADDQ  $4, BX
	DECQ  CX
	JNZ  pc_ur_sse2_copy_scalar

pc_ur_sse2_rows:
	MOVQ predictionError+16(FP), AX

pc_ur_sse2_row:
	TESTQ R9, R9
	JZ   pc_ur_sse2_done

	MOVSS (AX), X0
	CVTSS2SD X0, X6
	CVTSS2SD X15, X7
	MULSD X7, X6

	MOVQ R11, SI
	MOVQ DI, BX
	MOVQ R8, CX

pc_ur_sse2_w4:
	CMPQ CX, $4
	JL   pc_ur_sse2_tail

	MOVQ $4, DX

pc_ur_sse2_w4_each:
	MOVSS (SI), X1
	CVTSS2SD X1, X2
	MULSD X6, X2
	CVTSD2SS X2, X2
	MOVSS (BX), X3
	ADDSS X2, X3
	MOVSS X3, (BX)
	ADDQ  $4, SI
	ADDQ  $4, BX
	DECQ  DX
	JNZ  pc_ur_sse2_w4_each

	SUBQ $4, CX
	JMP  pc_ur_sse2_w4

pc_ur_sse2_tail:
	TESTQ CX, CX
	JZ   pc_ur_sse2_next_row

pc_ur_sse2_scalar:
	MOVSS (SI), X1
	CVTSS2SD X1, X2
	MULSD X6, X2
	CVTSD2SS X2, X2
	MOVSS (BX), X3
	ADDSS X2, X3
	MOVSS X3, (BX)
	ADDQ  $4, SI
	ADDQ  $4, BX
	DECQ  CX
	JNZ  pc_ur_sse2_scalar

pc_ur_sse2_next_row:
	ADDQ $4, AX
	MOVQ R8, CX
	SHLQ $2, CX
	ADDQ CX, R11
	DECQ R9
	JMP  pc_ur_sse2_row

pc_ur_sse2_done:
	RET

// func PCUpdateWeightsFloat32SSE2Asm(weights, representation, predictionError, output *float32, learningRate float32, outDim, inDim int)
TEXT ·PCUpdateWeightsFloat32SSE2Asm(SB), NOSPLIT, $0-56
	MOVQ weights+0(FP), R11
	MOVQ representation+8(FP), R12
	MOVQ predictionError+16(FP), R10
	MOVQ output+24(FP), DI
	MOVSS learningRate+32(FP), X15
	MOVQ outDim+40(FP), R9
	MOVQ inDim+48(FP), R8

	MOVQ R11, SI
	MOVQ DI, BX
	MOVQ R9, CX
	IMULQ R8, CX

pc_uw_sse2_copy_w4:
	CMPQ CX, $4
	JL   pc_uw_sse2_copy_tail

	VMOVUPS (SI), X0
	VMOVUPS X0, (BX)

	ADDQ $16, SI
	ADDQ $16, BX
	SUBQ $4, CX
	JMP  pc_uw_sse2_copy_w4

pc_uw_sse2_copy_tail:
	TESTQ CX, CX
	JZ   pc_uw_sse2_rows

pc_uw_sse2_copy_scalar:
	MOVSS (SI), X0
	MOVSS X0, (BX)
	ADDQ  $4, SI
	ADDQ  $4, BX
	DECQ  CX
	JNZ  pc_uw_sse2_copy_scalar

pc_uw_sse2_rows:
pc_uw_sse2_row:
	TESTQ R9, R9
	JZ   pc_uw_sse2_done

	MOVSS (R10), X0
	CVTSS2SD X0, X6
	CVTSS2SD X15, X7
	MULSD X7, X6

	MOVQ R11, SI
	MOVQ R12, DX
	MOVQ DI, BX
	MOVQ R8, CX

pc_uw_sse2_w4:
	CMPQ CX, $4
	JL   pc_uw_sse2_tail

	MOVQ $4, R13

pc_uw_sse2_w4_each:
	MOVSS (DX), X1
	CVTSS2SD X1, X2
	MULSD X6, X2
	CVTSD2SS X2, X2
	MOVSS (BX), X3
	ADDSS X2, X3
	MOVSS X3, (BX)
	ADDQ  $4, DX
	ADDQ  $4, BX
	DECQ  R13
	JNZ  pc_uw_sse2_w4_each

	SUBQ $4, CX
	JMP  pc_uw_sse2_w4

pc_uw_sse2_tail:
	TESTQ CX, CX
	JZ   pc_uw_sse2_next_row

pc_uw_sse2_scalar:
	MOVSS (DX), X1
	CVTSS2SD X1, X2
	MULSD X6, X2
	CVTSD2SS X2, X2
	MOVSS (BX), X3
	ADDSS X2, X3
	MOVSS X3, (BX)
	ADDQ  $4, DX
	ADDQ  $4, BX
	DECQ  CX
	JNZ  pc_uw_sse2_scalar

pc_uw_sse2_next_row:
	ADDQ $4, R10
	MOVQ R8, AX
	SHLQ $2, AX
	ADDQ AX, R11
	ADDQ AX, DI
	DECQ R9
	JMP  pc_uw_sse2_row

pc_uw_sse2_done:
	RET
