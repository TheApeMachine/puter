#include "textflag.h"

DATA maskZeroAVX2<>+0(SB)/4, $0.0
DATA maskZeroAVX2<>+4(SB)/4, $0.0
DATA maskZeroAVX2<>+8(SB)/4, $0.0
DATA maskZeroAVX2<>+12(SB)/4, $0.0
GLOBL maskZeroAVX2<>(SB), RODATA|NOPTR, $16

DATA maskNegInfAVX2<>+0(SB)/4, $0xFF800000
DATA maskNegInfAVX2<>+4(SB)/4, $0xFF800000
DATA maskNegInfAVX2<>+8(SB)/4, $0xFF800000
DATA maskNegInfAVX2<>+12(SB)/4, $0xFF800000
GLOBL maskNegInfAVX2<>(SB), RODATA|NOPTR, $16

// func ApplyMaskFloat32AVX2Asm(input, mask, output *float32, count int)
TEXT ·ApplyMaskFloat32AVX2Asm(SB), NOSPLIT, $0-32
	MOVQ input+0(FP), DI
	MOVQ mask+8(FP), SI
	MOVQ output+16(FP), R8
	MOVQ count+24(FP), CX

mask_avx2_w8:
	CMPQ CX, $8
	JL   mask_avx2_w4

	VMOVUPS (DI), Y0
	VMOVUPS (SI), Y1
	VADDPS Y1, Y0, Y0
	VMOVUPS Y0, (R8)

	ADDQ $32, DI
	ADDQ $32, SI
	ADDQ $32, R8
	SUBQ $8, CX
	JMP  mask_avx2_w8

mask_avx2_w4:
	CMPQ CX, $4
	JL   mask_avx2_tail

	VMOVUPS (DI), X0
	VMOVUPS (SI), X1
	VADDPS X1, X0, X0
	VMOVUPS X0, (R8)

	ADDQ $16, DI
	ADDQ $16, SI
	ADDQ $16, R8
	SUBQ $4, CX
	JMP  mask_avx2_w4

mask_avx2_tail:
	TESTQ CX, CX
	JZ   mask_avx2_done

mask_avx2_scalar:
	VMOVSS (DI), X0
	VMOVSS (SI), X1
	VADDSS X1, X0, X0
	MOVSS X0, (R8)
	ADDQ $4, DI
	ADDQ $4, SI
	ADDQ $4, R8
	DECQ CX
	JNZ  mask_avx2_scalar

mask_avx2_done:
	RET

// func CausalMaskFloat32AVX2Asm(output *float32, seqQ, seqK int)
TEXT ·CausalMaskFloat32AVX2Asm(SB), NOSPLIT, $0-24
	MOVQ output+0(FP), DI
	MOVQ seqQ+8(FP), R10
	MOVQ seqK+16(FP), BX

	VBROADCASTSS maskZeroAVX2<>(SB), Y0
	VBROADCASTSS maskNegInfAVX2<>(SB), Y1
	VBROADCASTSS maskZeroAVX2<>(SB), X2
	VBROADCASTSS maskNegInfAVX2<>(SB), X3

	XORQ R11, R11

causal_avx2_row:
	CMPQ R11, R10
	JGE  causal_avx2_done

	MOVQ R11, AX
	INCQ AX
	CMPQ AX, BX
	JLE  causal_avx2_zero_len_ok
	MOVQ BX, AX

causal_avx2_zero_len_ok:
	MOVQ AX, CX

causal_avx2_zero_w8:
	CMPQ CX, $8
	JL   causal_avx2_zero_w4

	VMOVUPS Y0, (DI)

	ADDQ $32, DI
	SUBQ $8, CX
	JMP  causal_avx2_zero_w8

causal_avx2_zero_w4:
	CMPQ CX, $4
	JL   causal_avx2_zero_tail

	VMOVUPS X2, (DI)

	ADDQ $16, DI
	SUBQ $4, CX
	JMP  causal_avx2_zero_w4

causal_avx2_zero_tail:
	TESTQ CX, CX
	JZ   causal_avx2_zero_done

causal_avx2_zero_scalar:
	MOVSS X2, (DI)
	ADDQ $4, DI
	DECQ CX
	JNZ  causal_avx2_zero_scalar

causal_avx2_zero_done:
	MOVQ seqK+16(FP), BX
	MOVQ R11, AX
	INCQ AX
	CMPQ AX, BX
	JGE  causal_avx2_next_row

	MOVQ BX, R12
	SUBQ AX, R12
	MOVQ R12, CX

causal_avx2_inf_w8:
	CMPQ CX, $8
	JL   causal_avx2_inf_w4

	VMOVUPS Y1, (DI)

	ADDQ $32, DI
	SUBQ $8, CX
	JMP  causal_avx2_inf_w8

causal_avx2_inf_w4:
	CMPQ CX, $4
	JL   causal_avx2_inf_tail

	VMOVUPS X3, (DI)

	ADDQ $16, DI
	SUBQ $4, CX
	JMP  causal_avx2_inf_w4

causal_avx2_inf_tail:
	TESTQ CX, CX
	JZ   causal_avx2_next_row

causal_avx2_inf_scalar:
	MOVSS X3, (DI)
	ADDQ $4, DI
	DECQ CX
	JNZ  causal_avx2_inf_scalar

causal_avx2_next_row:
	INCQ R11
	JMP  causal_avx2_row

causal_avx2_done:
	RET

DATA maskIota8AVX2<>+0(SB)/4, $0
DATA maskIota8AVX2<>+4(SB)/4, $1
DATA maskIota8AVX2<>+8(SB)/4, $2
DATA maskIota8AVX2<>+12(SB)/4, $3
DATA maskIota8AVX2<>+16(SB)/4, $4
DATA maskIota8AVX2<>+20(SB)/4, $5
DATA maskIota8AVX2<>+24(SB)/4, $6
DATA maskIota8AVX2<>+28(SB)/4, $7
GLOBL maskIota8AVX2<>(SB), RODATA|NOPTR, $32

DATA maskIota4AVX2<>+0(SB)/4, $0
DATA maskIota4AVX2<>+4(SB)/4, $1
DATA maskIota4AVX2<>+8(SB)/4, $2
DATA maskIota4AVX2<>+12(SB)/4, $3
GLOBL maskIota4AVX2<>(SB), RODATA|NOPTR, $16

// func ALiBiBiasFloat32AVX2Asm(scores, slope, output *float32, seqQ, seqK int)
TEXT ·ALiBiBiasFloat32AVX2Asm(SB), NOSPLIT, $0-40
	MOVQ scores+0(FP), SI
	MOVQ slope+8(FP), R9
	MOVQ output+16(FP), DI
	MOVQ seqQ+24(FP), R10
	MOVQ seqK+32(FP), BX

	XORQ R11, R11

alibi_avx2_row:
	CMPQ R11, R10
	JGE  alibi_avx2_done

	VBROADCASTSS (R9), Y15
	VBROADCASTSS (R9), X15
	VMOVD R11, X13
	VPBROADCASTD X13, Y13

	XORQ R12, R12

alibi_avx2_col:
	MOVQ BX, CX
	SUBQ R12, CX
	JZ   alibi_avx2_row_done

	CMPQ CX, $8
	JL   alibi_avx2_col_w4

	VMOVUPS (SI), Y0
	VMOVD R12, X10
	VPBROADCASTD X10, Y12
	VPADDD maskIota8AVX2<>(SB), Y12, Y12
	VPSUBD Y12, Y13, Y11
	VCVTDQ2PS Y11, Y10
	VXORPS Y9, Y9, Y9
	VCMPPS $1, Y10, Y9, Y8
	VMULPS Y15, Y10, Y16
	VSUBPS Y16, Y0, Y1
	VBLENDVPS Y0, Y1, Y8, Y0
	VMOVUPS Y0, (DI)

	ADDQ $32, SI
	ADDQ $32, DI
	ADDQ $8, R12
	JMP  alibi_avx2_col

alibi_avx2_col_w4:
	CMPQ CX, $4
	JL   alibi_avx2_col_tail

	VMOVUPS (SI), X0
	VMOVD R12, X10
	VPBROADCASTD X10, X11
	VPADDD maskIota4AVX2<>(SB), X11, X11
	VPSUBD X11, X13, X12
	VCVTDQ2PS X12, X10
	VXORPS X9, X9, X9
	VCMPPS $1, X10, X9, X8
	VMULPS X15, X10, X11
	VSUBPS X11, X0, X1
	VBLENDVPS X0, X1, X8, X0
	VMOVUPS X0, (DI)

	ADDQ $16, SI
	ADDQ $16, DI
	ADDQ $4, R12
	JMP  alibi_avx2_col

alibi_avx2_col_tail:
	TESTQ CX, CX
	JZ   alibi_avx2_row_done

alibi_avx2_col_scalar:
	VMOVSS (SI), X0
	MOVQ R11, AX
	MOVQ R12, DX
	SUBQ DX, AX
	CMPQ AX, $0
	JL   alibi_avx2_keep_score

	CVTSL2SS AX, X1
	VMULSS X15, X1, X1
	VSUBSS X1, X0, X0

alibi_avx2_keep_score:
	MOVSS X0, (DI)
	ADDQ $4, SI
	ADDQ $4, DI
	INCQ R12
	DECQ CX
	JNZ  alibi_avx2_col_scalar

alibi_avx2_row_done:
	INCQ R11
	JMP  alibi_avx2_row

alibi_avx2_done:
	RET
