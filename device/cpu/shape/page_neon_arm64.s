// SPDX-License-Identifier: Apache-2.0
// NEON page write/gather kernels for float32 and 16-bit storage.
#include "textflag.h"

// func PageWriteFloat32NEONAsm(storage, values *float32, pageIDs, offsets *int32, out *float32, pageCount, pageSize, inner, valueRows int)
TEXT ·PageWriteFloat32NEONAsm(SB), NOSPLIT, $0-72
	MOVD storage+0(FP), R0
	MOVD values+8(FP), R1
	MOVD pageIDs+16(FP), R2
	MOVD offsets+24(FP), R3
	MOVD out+32(FP), R4
	MOVD pageCount+40(FP), R5
	MOVD pageSize+48(FP), R6
	MOVD inner+56(FP), R7
	MOVD valueRows+64(FP), R8

	MUL R6, R5, R9
	MUL R7, R9, R9
	MOVD R0, R10
	MOVD R4, R11

page_write_f32_neon_copy16:
	CMP  $16, R9
	BLT  page_write_f32_neon_copy4
	VLD1 (R10), [V0.S4, V1.S4, V2.S4, V3.S4]
	VST1 [V0.S4, V1.S4, V2.S4, V3.S4], (R11)
	ADD  $64, R10
	ADD  $64, R11
	SUB  $16, R9
	B    page_write_f32_neon_copy16

page_write_f32_neon_copy4:
	CMP  $4, R9
	BLT  page_write_f32_neon_copy_tail
	VLD1 (R10), [V0.S4]
	VST1 [V0.S4], (R11)
	ADD  $16, R10
	ADD  $16, R11
	SUB  $4, R9
	B    page_write_f32_neon_copy4

page_write_f32_neon_copy_tail:
	CBZ  R9, page_write_f32_neon_rows
	FMOVS (R10), F0
	FMOVS F0, (R11)
	ADD   $4, R10
	ADD   $4, R11
	SUB   $1, R9
	B     page_write_f32_neon_copy_tail

page_write_f32_neon_rows:
	CBZ R8, page_write_f32_neon_done
	MOVW  (R2), R12
	MOVW  (R3), R13
	MUL   R6, R12, R12
	ADD   R13, R12, R12
	MUL   R7, R12, R12
	LSL   $2, R12, R12
	ADD   R12, R4, R14
	MOVD  R1, R15
	MOVD  R7, R9

page_write_f32_neon_row4:
	CMP  $4, R9
	BLT  page_write_f32_neon_row_tail
	VLD1 (R15), [V0.S4]
	VST1 [V0.S4], (R14)
	ADD  $16, R15
	ADD  $16, R14
	SUB  $4, R9
	B    page_write_f32_neon_row4

page_write_f32_neon_row_tail:
	CBZ  R9, page_write_f32_neon_next_row
	FMOVS (R15), F0
	FMOVS F0, (R14)
	ADD   $4, R15
	ADD   $4, R14
	SUB   $1, R9
	B     page_write_f32_neon_row_tail

page_write_f32_neon_next_row:
	LSL  $2, R7, R12
	ADD  R12, R1, R1
	ADD  $4, R2
	ADD  $4, R3
	SUB  $1, R8
	B    page_write_f32_neon_rows

page_write_f32_neon_done:
	RET

// func PageGatherFloat32NEONAsm(storage *float32, pageTable *int32, out *float32, pageCount, pageSize, inner, outRows int)
TEXT ·PageGatherFloat32NEONAsm(SB), NOSPLIT, $0-56
	MOVD storage+0(FP), R0
	MOVD pageTable+8(FP), R1
	MOVD out+16(FP), R2
	MOVD pageSize+32(FP), R4
	MOVD inner+40(FP), R5
	MOVD outRows+48(FP), R6
	MOVD $0, R7

page_gather_f32_neon_rows:
	CBZ R6, page_gather_f32_neon_done
	MOVW (R1), R8
	MUL  R4, R8, R8
	ADD  R7, R8, R8
	MUL  R5, R8, R8
	LSL  $2, R8, R8
	ADD  R8, R0, R9
	MOVD R2, R10
	MOVD R5, R11

page_gather_f32_neon_row4:
	CMP  $4, R11
	BLT  page_gather_f32_neon_row_tail
	VLD1 (R9), [V0.S4]
	VST1 [V0.S4], (R10)
	ADD  $16, R9
	ADD  $16, R10
	SUB  $4, R11
	B    page_gather_f32_neon_row4

page_gather_f32_neon_row_tail:
	CBZ  R11, page_gather_f32_neon_next_row
	FMOVS (R9), F0
	FMOVS F0, (R10)
	ADD   $4, R9
	ADD   $4, R10
	SUB   $1, R11
	B     page_gather_f32_neon_row_tail

page_gather_f32_neon_next_row:
	LSL $2, R5, R8
	ADD R8, R2, R2
	ADD $1, R7
	CMP R4, R7
	BLT page_gather_f32_neon_same_page
	MOVD $0, R7
	ADD  $4, R1
page_gather_f32_neon_same_page:
	SUB $1, R6
	B   page_gather_f32_neon_rows

page_gather_f32_neon_done:
	RET

// func PageWriteUint16NEONAsm(storage, values *uint16, pageIDs, offsets *int32, out *uint16, pageCount, pageSize, inner, valueRows int)
TEXT ·PageWriteUint16NEONAsm(SB), NOSPLIT, $0-72
	MOVD storage+0(FP), R0
	MOVD values+8(FP), R1
	MOVD pageIDs+16(FP), R2
	MOVD offsets+24(FP), R3
	MOVD out+32(FP), R4
	MOVD pageCount+40(FP), R5
	MOVD pageSize+48(FP), R6
	MOVD inner+56(FP), R7
	MOVD valueRows+64(FP), R8

	MUL R6, R5, R9
	MUL R7, R9, R9
	MOVD R0, R10
	MOVD R4, R11

page_write_u16_neon_copy32:
	CMP  $32, R9
	BLT  page_write_u16_neon_copy8
	VLD1 (R10), [V0.H8, V1.H8, V2.H8, V3.H8]
	VST1 [V0.H8, V1.H8, V2.H8, V3.H8], (R11)
	ADD  $64, R10
	ADD  $64, R11
	SUB  $32, R9
	B    page_write_u16_neon_copy32

page_write_u16_neon_copy8:
	CMP  $8, R9
	BLT  page_write_u16_neon_copy_tail
	VLD1 (R10), [V0.H8]
	VST1 [V0.H8], (R11)
	ADD  $16, R10
	ADD  $16, R11
	SUB  $8, R9
	B    page_write_u16_neon_copy8

page_write_u16_neon_copy_tail:
	CBZ R9, page_write_u16_neon_rows
	MOVH (R10), R12
	MOVH R12, (R11)
	ADD  $2, R10
	ADD  $2, R11
	SUB  $1, R9
	B    page_write_u16_neon_copy_tail

page_write_u16_neon_rows:
	CBZ R8, page_write_u16_neon_done
	MOVW (R2), R12
	MOVW (R3), R13
	MUL  R6, R12, R12
	ADD  R13, R12, R12
	MUL  R7, R12, R12
	LSL  $1, R12, R12
	ADD  R12, R4, R14
	MOVD R1, R15
	MOVD R7, R9

page_write_u16_neon_row8:
	CMP  $8, R9
	BLT  page_write_u16_neon_row_tail
	VLD1 (R15), [V0.H8]
	VST1 [V0.H8], (R14)
	ADD  $16, R15
	ADD  $16, R14
	SUB  $8, R9
	B    page_write_u16_neon_row8

page_write_u16_neon_row_tail:
	CBZ R9, page_write_u16_neon_next_row
	MOVH (R15), R12
	MOVH R12, (R14)
	ADD  $2, R15
	ADD  $2, R14
	SUB  $1, R9
	B    page_write_u16_neon_row_tail

page_write_u16_neon_next_row:
	LSL $1, R7, R12
	ADD R12, R1, R1
	ADD $4, R2
	ADD $4, R3
	SUB $1, R8
	B   page_write_u16_neon_rows

page_write_u16_neon_done:
	RET

// func PageGatherUint16NEONAsm(storage *uint16, pageTable *int32, out *uint16, pageCount, pageSize, inner, outRows int)
TEXT ·PageGatherUint16NEONAsm(SB), NOSPLIT, $0-56
	MOVD storage+0(FP), R0
	MOVD pageTable+8(FP), R1
	MOVD out+16(FP), R2
	MOVD pageSize+32(FP), R4
	MOVD inner+40(FP), R5
	MOVD outRows+48(FP), R6
	MOVD $0, R7

page_gather_u16_neon_rows:
	CBZ R6, page_gather_u16_neon_done
	MOVW (R1), R8
	MUL  R4, R8, R8
	ADD  R7, R8, R8
	MUL  R5, R8, R8
	LSL  $1, R8, R8
	ADD  R8, R0, R9
	MOVD R2, R10
	MOVD R5, R11

page_gather_u16_neon_row8:
	CMP  $8, R11
	BLT  page_gather_u16_neon_row_tail
	VLD1 (R9), [V0.H8]
	VST1 [V0.H8], (R10)
	ADD  $16, R9
	ADD  $16, R10
	SUB  $8, R11
	B    page_gather_u16_neon_row8

page_gather_u16_neon_row_tail:
	CBZ R11, page_gather_u16_neon_next_row
	MOVH (R9), R8
	MOVH R8, (R10)
	ADD  $2, R9
	ADD  $2, R10
	SUB  $1, R11
	B    page_gather_u16_neon_row_tail

page_gather_u16_neon_next_row:
	LSL $1, R5, R8
	ADD R8, R2, R2
	ADD $1, R7
	CMP R4, R7
	BLT page_gather_u16_neon_same_page
	MOVD $0, R7
	ADD  $4, R1
page_gather_u16_neon_same_page:
	SUB $1, R6
	B   page_gather_u16_neon_rows

page_gather_u16_neon_done:
	RET
