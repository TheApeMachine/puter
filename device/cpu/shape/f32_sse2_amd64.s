// SPDX-License-Identifier: Apache-2.0
// SSE2 float32 shape kernels: contiguous copy, where, masked fill.
#include "textflag.h"

DATA shapeMaskBit4SSE2<>+0(SB)/4, $1
DATA shapeMaskBit4SSE2<>+4(SB)/4, $2
DATA shapeMaskBit4SSE2<>+8(SB)/4, $4
DATA shapeMaskBit4SSE2<>+12(SB)/4, $8
GLOBL shapeMaskBit4SSE2<>(SB), RODATA|NOPTR, $16

// func CopyContiguousFloat32SSE2Asm(dst, src *float32, count int)
TEXT ·CopyContiguousFloat32SSE2Asm(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ count+16(FP), CX

shape_sse2_copy_w4:
	CMPQ CX, $4
	JL   shape_sse2_copy_tail

	VMOVUPS (SI), X0
	VMOVUPS X0, (DI)

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  shape_sse2_copy_w4

shape_sse2_copy_tail:
	TESTQ CX, CX
	JZ   shape_sse2_copy_done

shape_sse2_copy_scalar:
	VMOVSS (SI), X0
	MOVSS X0, (DI)
	ADDQ $4, SI
	ADDQ $4, DI
	DECQ CX
	JNZ  shape_sse2_copy_scalar

shape_sse2_copy_done:
	RET

// func WhereFloat32SSE2Asm(dst, positive, negative *float32, mask *byte, count int)
TEXT ·WhereFloat32SSE2Asm(SB), NOSPLIT, $0-40
	MOVQ dst+0(FP), DI
	MOVQ positive+8(FP), SI
	MOVQ negative+16(FP), R8
	MOVQ mask+24(FP), R9
	MOVQ count+32(FP), R12
	XORQ R11, R11

shape_sse2_where_w4:
	CMPQ R12, $4
	JL   shape_sse2_where_tail

shape_sse2_where_w4_load:
	CMPQ R11, $0
	JNE  shape_sse2_where_w4_use

	MOVBQZX (R9), R10

shape_sse2_where_w4_use:
	MOVQ R10, AX
	MOVQ R11, BX
	MOVQ BX, CX
	SHRQ CL, AX
	ANDQ $0xFF, AX

	VMOVD AX, X10
	VPBROADCASTD X10, X10
	MOVOU shapeMaskBit4SSE2<>(SB), X14
	VPAND X10, X14, X11
	VXORPS X12, X12, X12
	VPCMPGTD X13, X11, X12

	VMOVUPS (SI), X0
	VMOVUPS (R8), X1
	VBLENDVPS X1, X0, X13, X0
	VMOVUPS X0, (DI)

	ADDQ $16, SI
	ADDQ $16, R8
	ADDQ $16, DI
	ADDQ $4, R11
	CMPQ R11, $8
	JL   shape_sse2_where_w4_no_advance

	XORQ R11, R11
	ADDQ $1, R9

shape_sse2_where_w4_no_advance:
	SUBQ $4, R12
	JMP  shape_sse2_where_w4

shape_sse2_where_tail:
	TESTQ R12, R12
	JZ   shape_sse2_where_done

shape_sse2_where_tail_load:
	CMPQ R11, $0
	JNE  shape_sse2_where_tail_use

	MOVBQZX (R9), R10

shape_sse2_where_tail_use:
	MOVQ R10, AX
	MOVQ R11, BX
	MOVQ BX, CX
	SHRQ CL, AX
	TESTQ $1, AX
	JZ   shape_sse2_where_tail_neg

	VMOVSS (SI), X0
	JMP  shape_sse2_where_tail_store

shape_sse2_where_tail_neg:
	VMOVSS (R8), X0

shape_sse2_where_tail_store:
	MOVSS X0, (DI)
	ADDQ $4, SI
	ADDQ $4, R8
	ADDQ $4, DI
	INCQ R11
	CMPQ R11, $8
	JL   shape_sse2_where_tail_continue

	XORQ R11, R11
	ADDQ $1, R9

shape_sse2_where_tail_continue:
	DECQ R12
	JNZ  shape_sse2_where_tail

shape_sse2_where_done:
	RET

// func MaskedFillFloat32SSE2Asm(dst, input *float32, fill float32, mask *byte, count int)
TEXT ·MaskedFillFloat32SSE2Asm(SB), NOSPLIT, $0-36
	MOVQ dst+0(FP), DI
	MOVQ input+8(FP), SI
	MOVSS fill+16(FP), X0
	MOVQ mask+24(FP), R9
	MOVQ count+32(FP), R12
	XORQ R11, R11

shape_sse2_mfill_w4:
	CMPQ R12, $4
	JL   shape_sse2_mfill_tail

shape_sse2_mfill_w4_load:
	CMPQ R11, $0
	JNE  shape_sse2_mfill_w4_use

	MOVBQZX (R9), R10

shape_sse2_mfill_w4_use:
	MOVQ R10, AX
	MOVQ R11, BX
	MOVQ BX, CX
	SHRQ CL, AX
	ANDQ $0xFF, AX

	VMOVD AX, X10
	VPBROADCASTD X10, X10
	MOVOU shapeMaskBit4SSE2<>(SB), X14
	VPAND X10, X14, X11
	VXORPS X12, X12, X12
	VPCMPGTD X13, X11, X12

	VMOVUPS (SI), X1
	VBLENDVPS X1, X0, X13, X1
	VMOVUPS X1, (DI)

	ADDQ $16, SI
	ADDQ $16, DI
	ADDQ $4, R11
	CMPQ R11, $8
	JL   shape_sse2_mfill_w4_no_advance

	XORQ R11, R11
	ADDQ $1, R9

shape_sse2_mfill_w4_no_advance:
	SUBQ $4, R12
	JMP  shape_sse2_mfill_w4

shape_sse2_mfill_tail:
	TESTQ R12, R12
	JZ   shape_sse2_mfill_done

shape_sse2_mfill_tail_load:
	CMPQ R11, $0
	JNE  shape_sse2_mfill_tail_use

	MOVBQZX (R9), R10

shape_sse2_mfill_tail_use:
	MOVQ R10, AX
	MOVQ R11, BX
	MOVQ BX, CX
	SHRQ CL, AX
	TESTQ $1, AX
	JZ   shape_sse2_mfill_tail_keep

	MOVSS X0, X1
	JMP  shape_sse2_mfill_tail_store

shape_sse2_mfill_tail_keep:
	VMOVSS (SI), X1

shape_sse2_mfill_tail_store:
	MOVSS X1, (DI)
	ADDQ $4, SI
	ADDQ $4, DI
	INCQ R11
	CMPQ R11, $8
	JL   shape_sse2_mfill_tail_continue

	XORQ R11, R11
	ADDQ $1, R9

shape_sse2_mfill_tail_continue:
	DECQ R12
	JNZ  shape_sse2_mfill_tail

shape_sse2_mfill_done:
	RET
