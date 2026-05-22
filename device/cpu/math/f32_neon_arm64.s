// SPDX-License-Identifier: Apache-2.0
// NEON float32 math kernels: inv_sqrt_dim_scale, logsumexp row parts, outer.
#include "textflag.h"

#define VFADD_S4(m, n, d)  WORD $(0x4E20D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFSUB_S4(m, n, d)  WORD $(0x4EA0D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_S4(m, n, d)  WORD $(0x6E20DC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFMLA_S4(m, n, d)  WORD $(0x4E20CC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFDIV_S4(m, n, d)  WORD $(0x6E20FC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFRINTN_S4(n, d)   WORD $(0x4E218800 | ((n) << 5) | (d))
#define VFCVTZS_S4(n, d)   WORD $(0x4EA1B800 | ((n) << 5) | (d))
#define VADD_S4(m, n, d)   WORD $(0x4EA08400 | ((m) << 16) | ((n) << 5) | (d))
#define VSHL_S4_BY23(n, d) WORD $(0x4F375400 | ((n) << 5) | (d))
#define VMOV_B16(src, dst) WORD $(0x4EA01C00 | ((src) << 16) | ((src) << 5) | (dst))
#define VFMAX_S4(m, n, d)  WORD $(0x4E20F400 | ((m) << 16) | ((n) << 5) | (d))
#define VFADDP_S4(m, n, d) WORD $(0x6E30D400 | ((m) << 16) | ((n) << 5) | (d))
#define FADDP_S(n, d)      WORD $(0x7E30D800 | ((n) << 5) | (d))

DATA mathOneF32<>+0(SB)/4, $0x3f800000
GLOBL mathOneF32<>(SB), RODATA|NOPTR, $4

DATA mathExpC<>+0(SB)/4, $1.4426950408889634
DATA mathExpC<>+4(SB)/4, $0.6931471805599453
DATA mathExpC<>+8(SB)/4, $127.0
DATA mathExpC<>+12(SB)/4, $0.00019841270
DATA mathExpC<>+16(SB)/4, $0.0013888889
DATA mathExpC<>+20(SB)/4, $0.008333334
DATA mathExpC<>+24(SB)/4, $0.041666667
DATA mathExpC<>+28(SB)/4, $0.16666667
DATA mathExpC<>+32(SB)/4, $0.5
DATA mathExpC<>+36(SB)/4, $1.0
DATA mathExpC<>+40(SB)/4, $1.0
GLOBL mathExpC<>(SB), RODATA|NOPTR, $44

DATA mathSoftmaxClamp<>+0(SB)/4, $-87.0
GLOBL mathSoftmaxClamp<>(SB), RODATA|NOPTR, $4

#define MATH_EXP_V0_TO_V6 \
    VFMUL_S4(16, 0, 1) \
    VFRINTN_S4(1, 1) \
    VFMUL_S4(17, 1, 2) \
    VFSUB_S4(2, 0, 0) \
    VMOV_B16(19, 3) \
    VMOV_B16(20, 4) ; VFMLA_S4(0, 3, 4) \
    VMOV_B16(21, 3) ; VFMLA_S4(0, 4, 3) \
    VMOV_B16(22, 4) ; VFMLA_S4(0, 3, 4) \
    VMOV_B16(23, 3) ; VFMLA_S4(0, 4, 3) \
    VMOV_B16(24, 4) ; VFMLA_S4(0, 3, 4) \
    VMOV_B16(25, 3) ; VFMLA_S4(0, 4, 3) \
    VMOV_B16(26, 4) ; VFMLA_S4(0, 3, 4) \
    VFCVTZS_S4(1, 5) \
    VADD_S4(27, 5, 5) \
    VSHL_S4_BY23(5, 5) \
    VFMUL_S4(5, 4, 6)

// func InvSqrtDimScaleFloat32NEONAsm(out, input *float32, scale float32, count int)
TEXT ·InvSqrtDimScaleFloat32NEONAsm(SB), NOSPLIT, $0-28
	MOVD out+0(FP), R0
	MOVD input+8(FP), R1
	FMOVS scale+16(FP), F31
	VDUP V31.S[0], V31.S4
	MOVD count+20(FP), R2

math_inv_loop4:
	CMP  $4, R2
	BLT  math_inv_scalar_tail

	VLD1.P 16(R1), [V0.S4]
	VFMUL_S4(31, 0, 0)
	VST1.P [V0.S4], 16(R0)
	SUB  $4, R2
	B    math_inv_loop4

math_inv_scalar_tail:
	CBZ  R2, math_inv_done

math_inv_scalar_loop:
	FMOVS (R1), F0
	FMULS F31, F0, F0
	FMOVS F0, (R0)
	ADD  $4, R0
	ADD  $4, R1
	SUB  $1, R2
	CBNZ R2, math_inv_scalar_loop

math_inv_done:
	RET

// func LogSumExpRowPartsFloat32NEONAsm(row *float32, cols int, maximum, expSum *float32)
TEXT ·LogSumExpRowPartsFloat32NEONAsm(SB), NOSPLIT, $16-32
	MOVD row+0(FP), R0
	MOVD cols+8(FP), R1
	CBZ  R1, math_lse_zero

	FMOVS (R0), F16
	VDUP V16.S[0], V16.S4
	ADD  $4, R0
	SUB  $1, R1

math_lse_max_loop4:
	CMP  $4, R1
	BLT  math_lse_max_scalar

	VLD1.P 16(R0), [V0.S4]
	VFMAX_S4(0, 16, 16)
	SUB  $4, R1
	B    math_lse_max_loop4

math_lse_max_scalar:
	CBZ  R1, math_lse_max_done

math_lse_max_scalar_loop:
	FMOVS (R0), F0
	FCMPS F0, F16
	FCSELS GT, F16, F0, F16
	ADD  $4, R0
	SUB  $1, R1
	CBNZ R1, math_lse_max_scalar_loop

math_lse_max_done:
	MOVD row+0(FP), R0
	MOVD cols+8(FP), R1

	MOVD $mathExpC<>(SB), R3
	FMOVS 0(R3), F16
	FMOVS 4(R3), F17
	FMOVS 8(R3), F18
	FMOVS 12(R3), F19
	FMOVS 16(R3), F20
	FMOVS 20(R3), F21
	FMOVS 24(R3), F22
	FMOVS 28(R3), F23
	FMOVS 32(R3), F24
	FMOVS 36(R3), F25
	FMOVS 40(R3), F26
	VFCVTZS_S4(18, 27)
	FMOVS mathSoftmaxClamp<>(SB), F30
	VDUP V30.S[0], V30.S4
	FMOVS mathOneF32<>(SB), F29
	VDUP V29.S[0], V29.S4
	VEOR V31.B16, V31.B16, V31.B16

math_lse_exp_loop4:
	CMP  $4, R1
	BLT  math_lse_exp_scalar

	VLD1.P 16(R0), [V0.S4]
	VFSUB_S4(16, 0, 0)
	VFDIV_S4(29, 0, 0)
	VFMAX_S4(30, 0, 0)
	MATH_EXP_V0_TO_V6
	VFADD_S4(6, 31, 31)
	SUB  $4, R1
	B    math_lse_exp_loop4

math_lse_exp_scalar:
	VFADDP_S4(31, 31, 31)
	FADDP_S(31, 31)

	CBZ  R1, math_lse_store

math_lse_exp_scalar_loop:
	FMOVS (R0), F0
	FSUBS F16, F0, F0
	FDIVS F29, F0, F0
	FMAXS F30, F0, F0
	VDUP V0.S[0], V0.S4
	MATH_EXP_V0_TO_V6
	FADDS F6, F31, F31
	ADD  $4, R0
	SUB  $1, R1
	CBNZ R1, math_lse_exp_scalar_loop

math_lse_store:
	MOVD maximum+16(FP), R2
	MOVD expSum+24(FP), R3
	FMOVS F16, (R2)
	FMOVS F31, (R3)
	RET

math_lse_zero:
	MOVD maximum+16(FP), R2
	MOVD expSum+24(FP), R3
	FMOVS ZR, F0
	FMOVS F0, (R2)
	FMOVS F0, (R3)
	RET

// func OuterFloat32NEONAsm(out, left, right *float32, leftCount, rightCount int)
TEXT ·OuterFloat32NEONAsm(SB), NOSPLIT, $0-40
	MOVD out+0(FP), R0
	MOVD left+8(FP), R1
	MOVD right+16(FP), R2
	MOVD leftCount+24(FP), R3
	MOVD rightCount+32(FP), R4
	MOVD R4, R5
	LSL  $2, R5, R5

math_outer_row:
	CBZ  R3, math_outer_done

	FMOVS (R1), F31
	VDUP V31.S[0], V31.S4
	MOVD R2, R6
	MOVD R4, R7

math_outer_col_loop4:
	CMP  $4, R7
	BLT  math_outer_col_scalar

	VLD1.P 16(R6), [V0.S4]
	VFMUL_S4(31, 0, 0)
	VST1.P [V0.S4], 16(R0)
	SUB  $4, R7
	B    math_outer_col_loop4

math_outer_col_scalar:
	CBZ  R7, math_outer_next_row

math_outer_col_scalar_loop:
	FMOVS (R6), F0
	FMULS F31, F0, F0
	FMOVS F0, (R0)
	ADD  $4, R0
	ADD  $4, R6
	SUB  $1, R7
	CBNZ R7, math_outer_col_scalar_loop

math_outer_next_row:
	ADD  $4, R1
	ADD  R5, R0
	SUB  $1, R3
	B    math_outer_row

math_outer_done:
	RET
