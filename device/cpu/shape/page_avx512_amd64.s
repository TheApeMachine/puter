// SPDX-License-Identifier: Apache-2.0
// AVX-512 page write/gather kernels for float32 and 16-bit storage.
#include "textflag.h"

// func PageWriteFloat32AVX512Asm(storage, values *float32, pageIDs, offsets *int32, out *float32, pageCount, pageSize, inner, valueRows int)
TEXT ·PageWriteFloat32AVX512Asm(SB), NOSPLIT, $0-72
	MOVQ storage+0(FP), DI
	MOVQ values+8(FP), SI
	MOVQ pageIDs+16(FP), DX
	MOVQ offsets+24(FP), R8
	MOVQ out+32(FP), R9
	MOVQ pageCount+40(FP), R10
	MOVQ pageSize+48(FP), R11
	MOVQ inner+56(FP), R12
	MOVQ valueRows+64(FP), R13

	MOVQ R10, BX
	IMULQ R11, BX
	IMULQ R12, BX
	MOVQ DI, R14
	MOVQ R9, R15

page_write_f32_avx512_copy_w16:
	CMPQ BX, $16
	JL   page_write_f32_avx512_copy_w8
	VMOVUPS (R14), Z0
	VMOVUPS Z0, (R15)
	ADDQ $64, R14
	ADDQ $64, R15
	SUBQ $16, BX
	JMP  page_write_f32_avx512_copy_w16

page_write_f32_avx512_copy_w8:
	CMPQ BX, $8
	JL   page_write_f32_avx512_copy_w4
	VMOVUPS (R14), Y0
	VMOVUPS Y0, (R15)
	ADDQ $32, R14
	ADDQ $32, R15
	SUBQ $8, BX
	JMP  page_write_f32_avx512_copy_w8

page_write_f32_avx512_copy_w4:
	CMPQ BX, $4
	JL   page_write_f32_avx512_copy_tail
	VMOVUPS (R14), X0
	VMOVUPS X0, (R15)
	ADDQ $16, R14
	ADDQ $16, R15
	SUBQ $4, BX
	JMP  page_write_f32_avx512_copy_w4

page_write_f32_avx512_copy_tail:
	TESTQ BX, BX
	JZ    page_write_f32_avx512_rows
	VMOVSS (R14), X0
	MOVSS  X0, (R15)
	ADDQ   $4, R14
	ADDQ   $4, R15
	DECQ   BX
	JMP    page_write_f32_avx512_copy_tail

page_write_f32_avx512_rows:
	TESTQ R13, R13
	JZ    page_write_f32_avx512_done
	MOVLQSX (DX), AX
	MOVLQSX (R8), BX
	IMULQ   R11, AX
	ADDQ    BX, AX
	IMULQ   R12, AX
	LEAQ    (R9)(AX*4), R14
	MOVQ    SI, R15
	MOVQ    R12, BX

page_write_f32_avx512_row_w16:
	CMPQ BX, $16
	JL   page_write_f32_avx512_row_w8
	VMOVUPS (R15), Z0
	VMOVUPS Z0, (R14)
	ADDQ $64, R15
	ADDQ $64, R14
	SUBQ $16, BX
	JMP  page_write_f32_avx512_row_w16

page_write_f32_avx512_row_w8:
	CMPQ BX, $8
	JL   page_write_f32_avx512_row_w4
	VMOVUPS (R15), Y0
	VMOVUPS Y0, (R14)
	ADDQ $32, R15
	ADDQ $32, R14
	SUBQ $8, BX
	JMP  page_write_f32_avx512_row_w8

page_write_f32_avx512_row_w4:
	CMPQ BX, $4
	JL   page_write_f32_avx512_row_tail
	VMOVUPS (R15), X0
	VMOVUPS X0, (R14)
	ADDQ $16, R15
	ADDQ $16, R14
	SUBQ $4, BX
	JMP  page_write_f32_avx512_row_w4

page_write_f32_avx512_row_tail:
	TESTQ BX, BX
	JZ    page_write_f32_avx512_next_row
	VMOVSS (R15), X0
	MOVSS  X0, (R14)
	ADDQ   $4, R15
	ADDQ   $4, R14
	DECQ   BX
	JMP    page_write_f32_avx512_row_tail

page_write_f32_avx512_next_row:
	LEAQ (SI)(R12*4), SI
	ADDQ $4, DX
	ADDQ $4, R8
	DECQ R13
	JMP  page_write_f32_avx512_rows

page_write_f32_avx512_done:
	VZEROUPPER
	RET

// func PageGatherFloat32AVX512Asm(storage *float32, pageTable *int32, out *float32, pageCount, pageSize, inner, outRows int)
TEXT ·PageGatherFloat32AVX512Asm(SB), NOSPLIT, $0-56
	MOVQ storage+0(FP), DI
	MOVQ pageTable+8(FP), SI
	MOVQ out+16(FP), DX
	MOVQ pageSize+32(FP), R9
	MOVQ inner+40(FP), R10
	MOVQ outRows+48(FP), R11
	XORQ R12, R12

page_gather_f32_avx512_rows:
	TESTQ R11, R11
	JZ    page_gather_f32_avx512_done
	MOVLQSX (SI), AX
	IMULQ   R9, AX
	ADDQ    R12, AX
	IMULQ   R10, AX
	LEAQ    (DI)(AX*4), R14
	MOVQ    DX, R15
	MOVQ    R10, BX

page_gather_f32_avx512_row_w16:
	CMPQ BX, $16
	JL   page_gather_f32_avx512_row_w8
	VMOVUPS (R14), Z0
	VMOVUPS Z0, (R15)
	ADDQ $64, R14
	ADDQ $64, R15
	SUBQ $16, BX
	JMP  page_gather_f32_avx512_row_w16

page_gather_f32_avx512_row_w8:
	CMPQ BX, $8
	JL   page_gather_f32_avx512_row_w4
	VMOVUPS (R14), Y0
	VMOVUPS Y0, (R15)
	ADDQ $32, R14
	ADDQ $32, R15
	SUBQ $8, BX
	JMP  page_gather_f32_avx512_row_w8

page_gather_f32_avx512_row_w4:
	CMPQ BX, $4
	JL   page_gather_f32_avx512_row_tail
	VMOVUPS (R14), X0
	VMOVUPS X0, (R15)
	ADDQ $16, R14
	ADDQ $16, R15
	SUBQ $4, BX
	JMP  page_gather_f32_avx512_row_w4

page_gather_f32_avx512_row_tail:
	TESTQ BX, BX
	JZ    page_gather_f32_avx512_next_row
	VMOVSS (R14), X0
	MOVSS  X0, (R15)
	ADDQ   $4, R14
	ADDQ   $4, R15
	DECQ   BX
	JMP    page_gather_f32_avx512_row_tail

page_gather_f32_avx512_next_row:
	LEAQ (DX)(R10*4), DX
	INCQ R12
	CMPQ R12, R9
	JL   page_gather_f32_avx512_same_page
	XORQ R12, R12
	ADDQ $4, SI
page_gather_f32_avx512_same_page:
	DECQ R11
	JMP  page_gather_f32_avx512_rows

page_gather_f32_avx512_done:
	VZEROUPPER
	RET

// func PageWriteUint16AVX512Asm(storage, values *uint16, pageIDs, offsets *int32, out *uint16, pageCount, pageSize, inner, valueRows int)
TEXT ·PageWriteUint16AVX512Asm(SB), NOSPLIT, $0-72
	MOVQ storage+0(FP), DI
	MOVQ values+8(FP), SI
	MOVQ pageIDs+16(FP), DX
	MOVQ offsets+24(FP), R8
	MOVQ out+32(FP), R9
	MOVQ pageCount+40(FP), R10
	MOVQ pageSize+48(FP), R11
	MOVQ inner+56(FP), R12
	MOVQ valueRows+64(FP), R13

	MOVQ R10, BX
	IMULQ R11, BX
	IMULQ R12, BX
	MOVQ DI, R14
	MOVQ R9, R15

page_write_u16_avx512_copy_w32:
	CMPQ BX, $32
	JL   page_write_u16_avx512_copy_w16
	VMOVDQU64 (R14), Z0
	VMOVDQU64 Z0, (R15)
	ADDQ $64, R14
	ADDQ $64, R15
	SUBQ $32, BX
	JMP  page_write_u16_avx512_copy_w32

page_write_u16_avx512_copy_w16:
	CMPQ BX, $16
	JL   page_write_u16_avx512_copy_w8
	VMOVDQU (R14), Y0
	VMOVDQU Y0, (R15)
	ADDQ $32, R14
	ADDQ $32, R15
	SUBQ $16, BX
	JMP  page_write_u16_avx512_copy_w16

page_write_u16_avx512_copy_w8:
	CMPQ BX, $8
	JL   page_write_u16_avx512_copy_tail
	VMOVDQU (R14), X0
	VMOVDQU X0, (R15)
	ADDQ $16, R14
	ADDQ $16, R15
	SUBQ $8, BX
	JMP  page_write_u16_avx512_copy_w8

page_write_u16_avx512_copy_tail:
	TESTQ BX, BX
	JZ    page_write_u16_avx512_rows
	MOVW  (R14), AX
	MOVW  AX, (R15)
	ADDQ  $2, R14
	ADDQ  $2, R15
	DECQ  BX
	JMP   page_write_u16_avx512_copy_tail

page_write_u16_avx512_rows:
	TESTQ R13, R13
	JZ    page_write_u16_avx512_done
	MOVLQSX (DX), AX
	MOVLQSX (R8), BX
	IMULQ   R11, AX
	ADDQ    BX, AX
	IMULQ   R12, AX
	LEAQ    (R9)(AX*2), R14
	MOVQ    SI, R15
	MOVQ    R12, BX

page_write_u16_avx512_row_w32:
	CMPQ BX, $32
	JL   page_write_u16_avx512_row_w16
	VMOVDQU64 (R15), Z0
	VMOVDQU64 Z0, (R14)
	ADDQ $64, R15
	ADDQ $64, R14
	SUBQ $32, BX
	JMP  page_write_u16_avx512_row_w32

page_write_u16_avx512_row_w16:
	CMPQ BX, $16
	JL   page_write_u16_avx512_row_w8
	VMOVDQU (R15), Y0
	VMOVDQU Y0, (R14)
	ADDQ $32, R15
	ADDQ $32, R14
	SUBQ $16, BX
	JMP  page_write_u16_avx512_row_w16

page_write_u16_avx512_row_w8:
	CMPQ BX, $8
	JL   page_write_u16_avx512_row_tail
	VMOVDQU (R15), X0
	VMOVDQU X0, (R14)
	ADDQ $16, R15
	ADDQ $16, R14
	SUBQ $8, BX
	JMP  page_write_u16_avx512_row_w8

page_write_u16_avx512_row_tail:
	TESTQ BX, BX
	JZ    page_write_u16_avx512_next_row
	MOVW  (R15), AX
	MOVW  AX, (R14)
	ADDQ  $2, R15
	ADDQ  $2, R14
	DECQ  BX
	JMP   page_write_u16_avx512_row_tail

page_write_u16_avx512_next_row:
	LEAQ (SI)(R12*2), SI
	ADDQ $4, DX
	ADDQ $4, R8
	DECQ R13
	JMP  page_write_u16_avx512_rows

page_write_u16_avx512_done:
	VZEROUPPER
	RET

// func PageGatherUint16AVX512Asm(storage *uint16, pageTable *int32, out *uint16, pageCount, pageSize, inner, outRows int)
TEXT ·PageGatherUint16AVX512Asm(SB), NOSPLIT, $0-56
	MOVQ storage+0(FP), DI
	MOVQ pageTable+8(FP), SI
	MOVQ out+16(FP), DX
	MOVQ pageSize+32(FP), R9
	MOVQ inner+40(FP), R10
	MOVQ outRows+48(FP), R11
	XORQ R12, R12

page_gather_u16_avx512_rows:
	TESTQ R11, R11
	JZ    page_gather_u16_avx512_done
	MOVLQSX (SI), AX
	IMULQ   R9, AX
	ADDQ    R12, AX
	IMULQ   R10, AX
	LEAQ    (DI)(AX*2), R14
	MOVQ    DX, R15
	MOVQ    R10, BX

page_gather_u16_avx512_row_w32:
	CMPQ BX, $32
	JL   page_gather_u16_avx512_row_w16
	VMOVDQU64 (R14), Z0
	VMOVDQU64 Z0, (R15)
	ADDQ $64, R14
	ADDQ $64, R15
	SUBQ $32, BX
	JMP  page_gather_u16_avx512_row_w32

page_gather_u16_avx512_row_w16:
	CMPQ BX, $16
	JL   page_gather_u16_avx512_row_w8
	VMOVDQU (R14), Y0
	VMOVDQU Y0, (R15)
	ADDQ $32, R14
	ADDQ $32, R15
	SUBQ $16, BX
	JMP  page_gather_u16_avx512_row_w16

page_gather_u16_avx512_row_w8:
	CMPQ BX, $8
	JL   page_gather_u16_avx512_row_tail
	VMOVDQU (R14), X0
	VMOVDQU X0, (R15)
	ADDQ $16, R14
	ADDQ $16, R15
	SUBQ $8, BX
	JMP  page_gather_u16_avx512_row_w8

page_gather_u16_avx512_row_tail:
	TESTQ BX, BX
	JZ    page_gather_u16_avx512_next_row
	MOVW  (R14), AX
	MOVW  AX, (R15)
	ADDQ  $2, R14
	ADDQ  $2, R15
	DECQ  BX
	JMP   page_gather_u16_avx512_row_tail

page_gather_u16_avx512_next_row:
	LEAQ (DX)(R10*2), DX
	INCQ R12
	CMPQ R12, R9
	JL   page_gather_u16_avx512_same_page
	XORQ R12, R12
	ADDQ $4, SI
page_gather_u16_avx512_same_page:
	DECQ R11
	JMP  page_gather_u16_avx512_rows

page_gather_u16_avx512_done:
	VZEROUPPER
	RET
