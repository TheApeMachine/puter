// SPDX-License-Identifier: Apache-2.0
// AVX2 float32 shape kernels: contiguous copy, where, masked fill.
#include "textflag.h"

DATA shapeMaskBit8AVX2<>+0(SB)/4, $1
DATA shapeMaskBit8AVX2<>+4(SB)/4, $2
DATA shapeMaskBit8AVX2<>+8(SB)/4, $4
DATA shapeMaskBit8AVX2<>+12(SB)/4, $8
DATA shapeMaskBit8AVX2<>+16(SB)/4, $16
DATA shapeMaskBit8AVX2<>+20(SB)/4, $32
DATA shapeMaskBit8AVX2<>+24(SB)/4, $64
DATA shapeMaskBit8AVX2<>+28(SB)/4, $128
GLOBL shapeMaskBit8AVX2<>(SB), RODATA|NOPTR, $32

DATA shapeAllOnesAVX2<>+0(SB)/4, $-1
DATA shapeAllOnesAVX2<>+4(SB)/4, $-1
DATA shapeAllOnesAVX2<>+8(SB)/4, $-1
DATA shapeAllOnesAVX2<>+12(SB)/4, $-1
DATA shapeAllOnesAVX2<>+16(SB)/4, $-1
DATA shapeAllOnesAVX2<>+20(SB)/4, $-1
DATA shapeAllOnesAVX2<>+24(SB)/4, $-1
DATA shapeAllOnesAVX2<>+28(SB)/4, $-1
GLOBL shapeAllOnesAVX2<>(SB), RODATA|NOPTR, $32

// func CopyContiguousFloat32AVX2Asm(dst, src *float32, count int)
TEXT ·CopyContiguousFloat32AVX2Asm(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ count+16(FP), CX

shape_avx2_copy_w8:
	CMPQ CX, $8
	JL   shape_avx2_copy_w4

	VMOVUPS (SI), Y0
	VMOVUPS Y0, (DI)

	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $8, CX
	JMP  shape_avx2_copy_w8

shape_avx2_copy_w4:
	CMPQ CX, $4
	JL   shape_avx2_copy_tail

	VMOVUPS (SI), X0
	VMOVUPS X0, (DI)

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  shape_avx2_copy_w4

shape_avx2_copy_tail:
	TESTQ CX, CX
	JZ   shape_avx2_copy_done

shape_avx2_copy_scalar:
	VMOVSS (SI), X0
	MOVSS X0, (DI)
	ADDQ $4, SI
	ADDQ $4, DI
	DECQ CX
	JNZ  shape_avx2_copy_scalar

shape_avx2_copy_done:
	RET

// func WhereFloat32AVX2Asm(dst, positive, negative *float32, mask *byte, count int)
TEXT ·WhereFloat32AVX2Asm(SB), NOSPLIT, $0-40
	MOVQ dst+0(FP), DI
	MOVQ positive+8(FP), SI
	MOVQ negative+16(FP), R8
	MOVQ mask+24(FP), R9
	MOVQ count+32(FP), R12
	XORQ R11, R11

shape_avx2_where_w8:
	CMPQ R12, $8
	JL   shape_avx2_where_tail

shape_avx2_where_w8_load:
	CMPQ R11, $0
	JNE  shape_avx2_where_w8_use

	MOVBQZX (R9), R10

shape_avx2_where_w8_use:
	MOVQ R10, AX
	MOVQ R11, BX
	MOVQ BX, CX
	SHRQ CL, AX
	ANDQ $0xFF, AX

	VMOVD AX, X10
	VPBROADCASTD X10, Y10
	VMOVDQU shapeMaskBit8AVX2<>(SB), Y14
	VPAND Y10, Y14, Y11
	VPXOR Y12, Y12, Y12
	VPCMPEQD Y11, Y12, Y13
	VMOVDQU shapeAllOnesAVX2<>(SB), Y14
	VPANDN Y14, Y13, Y13

	VMOVUPS (SI), Y0
	VMOVUPS (R8), Y1
	VANDPS Y13, Y0, Y4
	VANDNPS Y1, Y13, Y5
	VORPS Y4, Y5, Y0
	VMOVUPS Y0, (DI)

	ADDQ $32, SI
	ADDQ $32, R8
	ADDQ $32, DI
	ADDQ $8, R11
	CMPQ R11, $8
	JL   shape_avx2_where_w8_no_advance

	XORQ R11, R11
	ADDQ $1, R9

shape_avx2_where_w8_no_advance:
	SUBQ $8, R12
	JMP  shape_avx2_where_w8

shape_avx2_where_tail:
	TESTQ R12, R12
	JZ   shape_avx2_where_done

shape_avx2_where_tail_load:
	CMPQ R11, $0
	JNE  shape_avx2_where_tail_use

	MOVBQZX (R9), R10

shape_avx2_where_tail_use:
	MOVQ R10, AX
	MOVQ R11, BX
	MOVQ BX, CX
	SHRQ CL, AX
	TESTQ $1, AX
	JZ   shape_avx2_where_tail_neg

	VMOVSS (SI), X0
	JMP  shape_avx2_where_tail_store

shape_avx2_where_tail_neg:
	VMOVSS (R8), X0

shape_avx2_where_tail_store:
	MOVSS X0, (DI)
	ADDQ $4, SI
	ADDQ $4, R8
	ADDQ $4, DI
	INCQ R11
	CMPQ R11, $8
	JL   shape_avx2_where_tail_continue

	XORQ R11, R11
	ADDQ $1, R9

shape_avx2_where_tail_continue:
	DECQ R12
	JNZ  shape_avx2_where_tail

shape_avx2_where_done:
	RET

// func MaskedFillFloat32AVX2Asm(dst, input *float32, fill float32, mask *byte, count int)
TEXT ·MaskedFillFloat32AVX2Asm(SB), NOSPLIT, $0-36
	MOVQ dst+0(FP), DI
	MOVQ input+8(FP), SI
	MOVSS fill+16(FP), X0
	MOVQ mask+24(FP), R9
	MOVQ count+32(FP), R12
	XORQ R11, R11

	VBROADCASTSS X0, Y15

shape_avx2_mfill_w8:
	CMPQ R12, $8
	JL   shape_avx2_mfill_tail

shape_avx2_mfill_w8_load:
	CMPQ R11, $0
	JNE  shape_avx2_mfill_w8_use

	MOVBQZX (R9), R10

shape_avx2_mfill_w8_use:
	MOVQ R10, AX
	MOVQ R11, BX
	MOVQ BX, CX
	SHRQ CL, AX
	ANDQ $0xFF, AX

	VMOVD AX, X10
	VPBROADCASTD X10, Y10
	VMOVDQU shapeMaskBit8AVX2<>(SB), Y14
	VPAND Y10, Y14, Y11
	VPXOR Y12, Y12, Y12
	VPCMPEQD Y11, Y12, Y13
	VMOVDQU shapeAllOnesAVX2<>(SB), Y14
	VPANDN Y14, Y13, Y13

	VMOVUPS (SI), Y0
	VANDPS Y13, Y15, Y4
	VANDNPS Y0, Y13, Y5
	VORPS Y4, Y5, Y0
	VMOVUPS Y0, (DI)

	ADDQ $32, SI
	ADDQ $32, DI
	ADDQ $8, R11
	CMPQ R11, $8
	JL   shape_avx2_mfill_w8_no_advance

	XORQ R11, R11
	ADDQ $1, R9

shape_avx2_mfill_w8_no_advance:
	SUBQ $8, R12
	JMP  shape_avx2_mfill_w8

shape_avx2_mfill_tail:
	TESTQ R12, R12
	JZ   shape_avx2_mfill_done

shape_avx2_mfill_tail_load:
	CMPQ R11, $0
	JNE  shape_avx2_mfill_tail_use

	MOVBQZX (R9), R10

shape_avx2_mfill_tail_use:
	MOVQ R10, AX
	MOVQ R11, BX
	MOVQ BX, CX
	SHRQ CL, AX
	TESTQ $1, AX
	JZ   shape_avx2_mfill_tail_keep

	MOVSS X0, X1
	JMP  shape_avx2_mfill_tail_store

shape_avx2_mfill_tail_keep:
	VMOVSS (SI), X1

shape_avx2_mfill_tail_store:
	MOVSS X1, (DI)
	ADDQ $4, SI
	ADDQ $4, DI
	INCQ R11
	CMPQ R11, $8
	JL   shape_avx2_mfill_tail_continue

	XORQ R11, R11
	ADDQ $1, R9

shape_avx2_mfill_tail_continue:
	DECQ R12
	JNZ  shape_avx2_mfill_tail

shape_avx2_mfill_done:
	RET
