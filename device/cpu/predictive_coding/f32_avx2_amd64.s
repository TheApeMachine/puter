#include "textflag.h"

// func PCPredictionErrorFloat32AVX2Asm(observed, predicted, output *float32, count int)
TEXT ·PCPredictionErrorFloat32AVX2Asm(SB), NOSPLIT, $0-28
	MOVQ observed+0(FP), SI
	MOVQ predicted+8(FP), DI
	MOVQ output+16(FP), BX
	MOVQ count+24(FP), CX

pc_pe_avx2_w8:
	CMPQ CX, $8
	JL   pc_pe_avx2_w4

	VMOVUPS (SI), Y0
	VMOVUPS (DI), Y1
	VSUBPS  Y1, Y0, Y0
	VMOVUPS Y0, (BX)

	ADDQ $32, SI
	ADDQ $32, DI
	ADDQ $32, BX
	SUBQ $8, CX
	JMP  pc_pe_avx2_w8

pc_pe_avx2_w4:
	CMPQ CX, $4
	JL   pc_pe_avx2_tail

	VMOVUPS (SI), X0
	VMOVUPS (DI), X1
	VSUBPS  X1, X0, X0
	VMOVUPS X0, (BX)

	ADDQ $16, SI
	ADDQ $16, DI
	ADDQ $16, BX
	SUBQ $4, CX
	JMP  pc_pe_avx2_w4

pc_pe_avx2_tail:
	TESTQ CX, CX
	JZ   pc_pe_avx2_done

pc_pe_avx2_scalar:
	MOVSS (SI), X0
	MOVSS (DI), X1
	SUBSS X1, X0
	MOVSS X0, (BX)
	ADDQ  $4, SI
	ADDQ  $4, DI
	ADDQ  $4, BX
	DECQ  CX
	JNZ  pc_pe_avx2_scalar

pc_pe_avx2_done:
	RET

// func PCPredictionFloat32AVX2Asm(weights, representation, output *float32, outDim, inDim int)
TEXT ·PCPredictionFloat32AVX2Asm(SB), NOSPLIT, $0-36
	MOVQ weights+0(FP), R11
	MOVQ representation+8(FP), R12
	MOVQ output+16(FP), DI
	MOVQ outDim+24(FP), R9
	MOVQ inDim+32(FP), R8

pc_pred_avx2_row:
	TESTQ R9, R9
	JZ   pc_pred_avx2_done

	MOVQ R11, SI
	MOVQ R12, DX
	MOVQ R8, CX

	VXORPD Y0, Y0, Y0

pc_pred_avx2_dot_w8:
	CMPQ CX, $8
	JL   pc_pred_avx2_dot_w4

	VMOVUPS (SI), Y1
	VMOVUPS (DX), Y2
	VEXTRACTF128 $0, Y1, X3
	VEXTRACTF128 $0, Y2, X4
	VCVTPS2PD X3, Y5
	VCVTPS2PD X4, Y6
	VFMADD231PD Y0, Y6, Y5
	VEXTRACTF128 $1, Y1, X3
	VEXTRACTF128 $1, Y2, X4
	VCVTPS2PD X3, Y5
	VCVTPS2PD X4, Y6
	VFMADD231PD Y0, Y6, Y5

	ADDQ $32, SI
	ADDQ $32, DX
	SUBQ $8, CX
	JMP  pc_pred_avx2_dot_w8

pc_pred_avx2_dot_w4:
	CMPQ CX, $4
	JL   pc_pred_avx2_dot_tail

	VMOVUPS (SI), X1
	VMOVUPS (DX), X2
	VCVTPS2PD X1, Y5
	VCVTPS2PD X2, Y6
	VFMADD231PD Y0, Y6, Y5
	MOVAPS X1, X3
	SHUFPS $0xEE, X1, X3
	MOVAPS X2, X4
	SHUFPS $0xEE, X2, X4
	VCVTPS2PD X3, Y5
	VCVTPS2PD X4, Y6
	VFMADD231PD Y0, Y6, Y5

	ADDQ $16, SI
	ADDQ $16, DX
	SUBQ $4, CX
	JMP  pc_pred_avx2_dot_w4

pc_pred_avx2_dot_tail:
	TESTQ CX, CX
	JZ   pc_pred_avx2_dot_reduce

pc_pred_avx2_dot_scalar:
	MOVSS (SI), X1
	MOVSS (DX), X2
	CVTSS2SD X1, X1
	CVTSS2SD X2, X2
	MULSD X2, X1
	ADDSD X1, X0
	ADDQ  $4, SI
	ADDQ  $4, DX
	DECQ  CX
	JNZ  pc_pred_avx2_dot_scalar

pc_pred_avx2_dot_reduce:
	VHADDPD Y1, Y0, Y0
	VHADDPD Y1, Y1, Y1
	VEXTRACTF128 $0, Y1, X0
	CVTSD2SS X0, X0
	MOVSS X0, (DI)

	ADDQ $4, DI
	MOVQ R8, AX
	SHLQ $2, AX
	ADDQ AX, R11
	DECQ R9
	JMP  pc_pred_avx2_row

pc_pred_avx2_done:
	RET

// func PCUpdateRepresentationFloat32AVX2Asm(weights, representation, predictionError, output *float32, learningRate float32, outDim, inDim int)
TEXT ·PCUpdateRepresentationFloat32AVX2Asm(SB), NOSPLIT, $0-56
	MOVQ weights+0(FP), R11
	MOVQ representation+8(FP), R12
	MOVQ predictionError+16(FP), R10
	MOVQ output+24(FP), DI
	MOVSS learningRate+32(FP), X15
	MOVQ outDim+40(FP), R9
	MOVQ inDim+48(FP), R8

	MOVQ R12, SI
	MOVQ DI, BX
	MOVQ R8, CX

pc_ur_avx2_copy_w8:
	CMPQ CX, $8
	JL   pc_ur_avx2_copy_w4

	VMOVUPS (SI), Y0
	VMOVUPS Y0, (BX)

	ADDQ $32, SI
	ADDQ $32, BX
	SUBQ $8, CX
	JMP  pc_ur_avx2_copy_w8

pc_ur_avx2_copy_w4:
	CMPQ CX, $4
	JL   pc_ur_avx2_copy_tail

	VMOVUPS (SI), X0
	VMOVUPS X0, (BX)

	ADDQ $16, SI
	ADDQ $16, BX
	SUBQ $4, CX
	JMP  pc_ur_avx2_copy_w4

pc_ur_avx2_copy_tail:
	TESTQ CX, CX
	JZ   pc_ur_avx2_rows

pc_ur_avx2_copy_scalar:
	MOVSS (SI), X0
	MOVSS X0, (BX)
	ADDQ  $4, SI
	ADDQ  $4, BX
	DECQ  CX
	JNZ  pc_ur_avx2_copy_scalar

pc_ur_avx2_rows:
pc_ur_avx2_row:
	TESTQ R9, R9
	JZ   pc_ur_avx2_done

	MOVSS (R10), X0
	VCVTSS2SD X0, X6, X6
	VCVTSS2SD X15, X7, X7
	VMULSD X7, X6, X6

	MOVQ R11, SI
	MOVQ DI, BX
	MOVQ R8, CX

pc_ur_avx2_w8:
	CMPQ CX, $8
	JL   pc_ur_avx2_w4

	MOVQ $8, DX

pc_ur_avx2_w8_each:
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
	JNZ  pc_ur_avx2_w8_each

	SUBQ $8, CX
	JMP  pc_ur_avx2_w8

pc_ur_avx2_w4:
	CMPQ CX, $4
	JL   pc_ur_avx2_tail

	MOVQ $4, DX

pc_ur_avx2_w4_each:
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
	JNZ  pc_ur_avx2_w4_each

	SUBQ $4, CX
	JMP  pc_ur_avx2_w4

pc_ur_avx2_tail:
	TESTQ CX, CX
	JZ   pc_ur_avx2_next_row

pc_ur_avx2_scalar:
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
	JNZ  pc_ur_avx2_scalar

pc_ur_avx2_next_row:
	ADDQ $4, R10
	MOVQ R8, AX
	SHLQ $2, AX
	ADDQ AX, R11
	DECQ R9
	JMP  pc_ur_avx2_row

pc_ur_avx2_done:
	RET

// func PCUpdateWeightsFloat32AVX2Asm(weights, representation, predictionError, output *float32, learningRate float32, outDim, inDim int)
TEXT ·PCUpdateWeightsFloat32AVX2Asm(SB), NOSPLIT, $0-56
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

pc_uw_avx2_copy_w8:
	CMPQ CX, $8
	JL   pc_uw_avx2_copy_w4

	VMOVUPS (SI), Y0
	VMOVUPS Y0, (BX)

	ADDQ $32, SI
	ADDQ $32, BX
	SUBQ $8, CX
	JMP  pc_uw_avx2_copy_w8

pc_uw_avx2_copy_w4:
	CMPQ CX, $4
	JL   pc_uw_avx2_copy_tail

	VMOVUPS (SI), X0
	VMOVUPS X0, (BX)

	ADDQ $16, SI
	ADDQ $16, BX
	SUBQ $4, CX
	JMP  pc_uw_avx2_copy_w4

pc_uw_avx2_copy_tail:
	TESTQ CX, CX
	JZ   pc_uw_avx2_rows

pc_uw_avx2_copy_scalar:
	MOVSS (SI), X0
	MOVSS X0, (BX)
	ADDQ  $4, SI
	ADDQ  $4, BX
	DECQ  CX
	JNZ  pc_uw_avx2_copy_scalar

pc_uw_avx2_rows:
pc_uw_avx2_row:
	TESTQ R9, R9
	JZ   pc_uw_avx2_done

	MOVSS (R10), X0
	VCVTSS2SD X0, X6, X6
	VCVTSS2SD X15, X7, X7
	VMULSD X7, X6, X6

	MOVQ R11, SI
	MOVQ R12, DX
	MOVQ DI, BX
	MOVQ R8, CX

pc_uw_avx2_w8:
	CMPQ CX, $8
	JL   pc_uw_avx2_w4

	MOVQ $8, R13

pc_uw_avx2_w8_each:
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
	JNZ  pc_uw_avx2_w8_each

	SUBQ $8, CX
	JMP  pc_uw_avx2_w8

pc_uw_avx2_w4:
	CMPQ CX, $4
	JL   pc_uw_avx2_tail

	MOVQ $4, R13

pc_uw_avx2_w4_each:
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
	JNZ  pc_uw_avx2_w4_each

	SUBQ $4, CX
	JMP  pc_uw_avx2_w4

pc_uw_avx2_tail:
	TESTQ CX, CX
	JZ   pc_uw_avx2_next_row

pc_uw_avx2_scalar:
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
	JNZ  pc_uw_avx2_scalar

pc_uw_avx2_next_row:
	ADDQ $4, R10
	MOVQ R8, AX
	SHLQ $2, AX
	ADDQ AX, R11
	ADDQ AX, DI
	DECQ R9
	JMP  pc_uw_avx2_row

pc_uw_avx2_done:
	RET
