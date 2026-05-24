// SPDX-License-Identifier: Apache-2.0
// NEON float32 sampling kernels: greedy argmax and temperature softmax row.
#include "textflag.h"

#define VFADD_S4(m, n, d)  WORD $(0x4E20D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFSUB_S4(m, n, d)  WORD $(0x4EA0D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_S4(m, n, d)  WORD $(0x6E20DC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFDIV_S4(m, n, d)  WORD $(0x6E20FC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFMLA_S4(m, n, d)  WORD $(0x4E20CC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFRINTN_S4(n, d)   WORD $(0x4E218800 | ((n) << 5) | (d))
#define VFCVTZS_S4(n, d)   WORD $(0x4EA1B800 | ((n) << 5) | (d))
#define VADD_S4(m, n, d)   WORD $(0x4EA08400 | ((m) << 16) | ((n) << 5) | (d))
#define VSHL_S4_BY23(n, d) WORD $(0x4F375400 | ((n) << 5) | (d))
#define VMOV_B16(src, dst) WORD $(0x4EA01C00 | ((src) << 16) | ((src) << 5) | (dst))
#define VFMAX_S4(m, n, d)  WORD $(0x4E20F400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMAXV_S4(n, d)    WORD $(0x6E30F800 | ((n) << 5) | (d))
#define VFCMEQ_S4(m, n, d) WORD $(0x6E20E400 | ((m) << 16) | ((n) << 5) | (d))
#define VFADDP_S4(m, n, d) WORD $(0x6E30D400 | ((m) << 16) | ((n) << 5) | (d))
#define FADDP_S(n, d)      WORD $(0x7E30D800 | ((n) << 5) | (d))

DATA samOneF32<>+0(SB)/4, $0x3f800000
GLOBL samOneF32<>(SB), RODATA|NOPTR, $4

#define VFCMLTZ_S4(n, d)   WORD $(0x4EA0E800 | ((n) << 5) | (d))
#define VFCMLT_S4(m, n, d) WORD $(0x6EA0E800 | ((m) << 16) | ((n) << 5) | (d))
#define VSCVTF_S4(n, d)    WORD $(0x4E21D800 | ((n) << 5) | (d))
#define VBIC_B16(m, n, d)  WORD $(0x4E601C00 | ((m) << 16) | ((n) << 5) | (d))

DATA samExpC<>+0(SB)/4, $1.4426950408889634
DATA samExpC<>+4(SB)/4, $0.6931471805599453
DATA samExpC<>+12(SB)/4, $0.69314718
DATA samExpC<>+16(SB)/4, $0.24022650
DATA samExpC<>+20(SB)/4, $0.05550410
DATA samExpC<>+24(SB)/4, $0.00961812
DATA samExpC<>+28(SB)/4, $0.00133389
DATA samExpC<>+32(SB)/4, $1.0
GLOBL samExpC<>(SB), RODATA|NOPTR, $36

DATA samSoftmaxClamp<>+0(SB)/4, $-87.33654
GLOBL samSoftmaxClamp<>(SB), RODATA|NOPTR, $4

#define SAM_ACCUM_V6_TO_F32_SUM \
	VFADD_S4(6, 29, 29)

#define SAM_EXP_V0_TO_V6 \
	VFMUL_S4(16, 0, 1) \
	VFCVTZS_S4(1, 5) \
	VFCMLTZ_S4(1, 7) \
	VADD_S4(7, 5, 5) \
	VSCVTF_S4(5, 2) \
	VFSUB_S4(2, 1, 0) \
	VMOV_B16(23, 3) \
	VMOV_B16(22, 4) \
	VFMLA_S4(0, 3, 4) \
	VMOV_B16(21, 3) \
	VFMLA_S4(0, 4, 3) \
	VMOV_B16(20, 4) \
	VFMLA_S4(0, 3, 4) \
	VMOV_B16(19, 3) \
	VFMLA_S4(0, 4, 3) \
	VMOV_B16(24, 4) \
	VFMLA_S4(0, 3, 4) \
	VSHL_S4_BY23(5, 5) \
	VADD_S4(5, 4, 6)

// func GreedySampleFloat32NEONAsm(logits *float32, count int) int32
TEXT ·GreedySampleFloat32NEONAsm(SB), NOSPLIT, $16-20
	MOVD logits+0(FP), R0
	MOVD R0, R10
	MOVD count+8(FP), R1
	CBZ  R1, sam_greedy_zero
	CMP  $1, R1
	BEQ  sam_greedy_one

	FMOVS (R0), F16
	VDUP V16.S[0], V16.S4
	ADD  $4, R0
	SUB  $1, R1

sam_greedy_max_loop4:
	CMP  $4, R1
	BLT  sam_greedy_max_scalar

	VLD1.P 16(R0), [V0.S4]
	VFMAX_S4(0, 16, 16)
	SUB  $4, R1
	B    sam_greedy_max_loop4

sam_greedy_max_scalar:
	VFMAXV_S4(16, 0)
	FMOVS F0, F16
	CBZ  R1, sam_greedy_max_done

sam_greedy_max_scalar_loop:
	FMOVS (R0), F0
	FCMPS F0, F16
	FCSELS GT, F16, F0, F16
	ADD  $4, R0
	SUB  $1, R1
	CBNZ R1, sam_greedy_max_scalar_loop

sam_greedy_max_done:
	MOVD R10, R0
	MOVD count+8(FP), R1
	VDUP V16.S[0], V16.S4
	MOVD $0, R8

sam_greedy_find_loop4:
	CMP  $4, R1
	BLT  sam_greedy_find_scalar

	VLD1.P 16(R0), [V0.S4]
	VFCMEQ_S4(16, 0, 1)
	VST1 [V1.S4], (RSP)
	MOVW (RSP), R2
	CBNZ R2, sam_greedy_found_lane0
	MOVW 4(RSP), R2
	CBNZ R2, sam_greedy_found_lane1
	MOVW 8(RSP), R2
	CBNZ R2, sam_greedy_found_lane2
	MOVW 12(RSP), R2
	CBNZ R2, sam_greedy_found_lane3
	ADD  $4, R8
	SUB  $4, R1
	B    sam_greedy_find_loop4

sam_greedy_found_lane0:
	MOVW R8, ret+16(FP)
	RET

sam_greedy_found_lane1:
	ADD  $1, R8
	MOVW R8, ret+16(FP)
	RET

sam_greedy_found_lane2:
	ADD  $2, R8
	MOVW R8, ret+16(FP)
	RET

sam_greedy_found_lane3:
	ADD  $3, R8
	MOVW R8, ret+16(FP)
	RET

sam_greedy_find_scalar:
	CBZ  R1, sam_greedy_fail

sam_greedy_find_scalar_loop:
	FMOVS (R0), F0
	FCMPS F0, F16
	BNE  sam_greedy_find_next
	MOVW R8, ret+16(FP)
	RET

sam_greedy_find_next:
	ADD  $4, R0
	ADD  $1, R8
	SUB  $1, R1
	CBNZ R1, sam_greedy_find_scalar_loop

sam_greedy_fail:
	MOVD count+8(FP), R0
	SUB  $1, R0
	MOVW R0, ret+16(FP)
	RET

sam_greedy_one:
	MOVW $0, ret+16(FP)
	RET

sam_greedy_zero:
	MOVW $0, ret+16(FP)
	RET

// func SamplingSoftmaxRowFloat32NEONAsm(logits, out *float32, temperature float32, count int)
TEXT ·SamplingSoftmaxRowFloat32NEONAsm(SB), NOSPLIT, $0-36
	MOVD logits+0(FP), R0
	MOVD out+8(FP), R1
	FMOVS temperature+16(FP), F10
	MOVD count+24(FP), R2
	CBZ  R2, sam_softmax_done

	FMOVS $0.0, F11
	FCMPS F10, F11
	BNE  sam_softmax_temp_ok
	FMOVS samOneF32<>(SB), F10

sam_softmax_temp_ok:
	VDUP V10.S[0], V10.S4

	FMOVS (R0), F16
	VDUP V16.S[0], V16.S4
	ADD  $4, R0
	SUB  $1, R2

sam_softmax_max_loop4:
	CMP  $4, R2
	BLT  sam_softmax_max_scalar

	VLD1 (R0), [V0.S4]
	ADD  $16, R0
	VFMAX_S4(0, 16, 16)
	SUB  $4, R2
	B    sam_softmax_max_loop4

sam_softmax_max_scalar:
	VFMAXV_S4(16, 0)
	FMOVS F0, F16
	CBZ  R2, sam_softmax_max_done

sam_softmax_max_scalar_loop:
	FMOVS (R0), F0
	FCMPS F0, F16
	FCSELS GT, F16, F0, F16
	ADD  $4, R0
	SUB  $1, R2
	CBNZ R2, sam_softmax_max_scalar_loop

sam_softmax_max_done:
	MOVD logits+0(FP), R0
	MOVD out+8(FP), R1
	MOVD count+24(FP), R2
	FMOVS F16, F28
	VDUP V28.S[0], V28.S4

	MOVD $samExpC<>(SB), R3
	FMOVS 0(R3), F16
	VDUP V16.S[0], V16.S4
	FMOVS 12(R3), F19
	VDUP V19.S[0], V19.S4
	FMOVS 16(R3), F20
	VDUP V20.S[0], V20.S4
	FMOVS 20(R3), F21
	VDUP V21.S[0], V21.S4
	FMOVS 24(R3), F22
	VDUP V22.S[0], V22.S4
	FMOVS 28(R3), F23
	VDUP V23.S[0], V23.S4
	FMOVS 32(R3), F24
	VDUP V24.S[0], V24.S4
	FMOVS samSoftmaxClamp<>(SB), F30
	VDUP V30.S[0], V30.S4
	VEOR V29.B16, V29.B16, V29.B16

sam_softmax_exp_loop4:
	CMP  $4, R2
	BLT  sam_softmax_exp_scalar

	VLD1 (R0), [V0.S4]
	ADD  $16, R0
	VFSUB_S4(28, 0, 0)
	VFDIV_S4(10, 0, 0)
	VFSUB_S4(30, 0, 8)
	VFCMLTZ_S4(8, 9)
	SAM_EXP_V0_TO_V6
	VBIC_B16(9, 6, 6)
	VST1 [V6.S4], (R1)
	SAM_ACCUM_V6_TO_F32_SUM
	ADD  $16, R1
	SUB  $4, R2
	B    sam_softmax_exp_loop4

sam_softmax_exp_scalar:
	VFADDP_S4(29, 29, 29)
	FADDP_S(29, 31)
	CBZ  R2, sam_softmax_normalize

	MOVD count+24(FP), R6
	SUB  R2, R6, R6
	LSL  $2, R6, R6
	MOVD logits+0(FP), R0
	ADD  R6, R0, R0
	MOVD out+8(FP), R1
	ADD  R6, R1, R1

sam_softmax_exp_scalar_loop:
	FMOVS (R0), F0
	FSUBS F28, F0, F0
	FDIVS F10, F0, F0
	VDUP V0.S[0], V0.S4
	SAM_EXP_V0_TO_V6

sam_softmax_exp_scalar_store:
	FMOVS F6, (R1)
	FADDS F6, F31, F31
	ADD  $4, R0
	ADD  $4, R1
	SUB  $1, R2
	CBNZ R2, sam_softmax_exp_scalar_loop

sam_softmax_normalize:
	FMOVS $0, F15
	FCMPS F31, F15
	BEQ  sam_softmax_done

	FMOVS samOneF32<>(SB), F8
	FDIVS F31, F8, F8
	VDUP V8.S[0], V8.S4

	MOVD out+8(FP), R0
	MOVD count+24(FP), R2

sam_softmax_scale_loop4:
	CMP  $4, R2
	BLT  sam_softmax_scale_scalar

	VLD1 (R0), [V0.S4]
	VFMUL_S4(8, 0, 0)
	VST1 [V0.S4], (R0)
	ADD  $16, R0
	SUB  $4, R2
	B    sam_softmax_scale_loop4

sam_softmax_scale_scalar:
	CBZ  R2, sam_softmax_done

sam_softmax_scale_scalar_loop:
	FMOVS (R0), F0
	FMULS F8, F0, F0
	FMOVS F0, (R0)
	ADD  $4, R0
	SUB  $1, R2
	CBNZ R2, sam_softmax_scale_scalar_loop

sam_softmax_done:
	RET
