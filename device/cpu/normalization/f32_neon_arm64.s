// SPDX-License-Identifier: Apache-2.0
// NEON float32 normalization kernels: squared diff sum and const scale/bias apply.
#include "textflag.h"

#define VFSUB_S4(m, n, d) WORD $(0x4EA0D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_S4(m, n, d) WORD $(0x6E20DC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFADD_S4(m, n, d) WORD $(0x4E20D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFADD_D2(m, n, d) WORD $(0x4E60D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_D2(m, n, d) WORD $(0x4E60FC00 | ((m) << 16) | ((n) << 5) | (d))
#define FCVTL_2D(n, d)    WORD $(0x0E617800 | ((n) << 5) | (d))
#define FCVTL2_2D(n, d)   WORD $(0x4E617800 | ((n) << 5) | (d))
#define FADDP_D(n, d)     WORD $(0x7E70D800 | ((n) << 5) | (d))

// func NormSquaredDiffSumFloat32NEONAsm(row *float32, count int, mean float32) float32
TEXT ·NormSquaredDiffSumFloat32NEONAsm(SB), NOSPLIT, $0-28
	MOVD row+0(FP), R0
	MOVD count+8(FP), R1
	FMOVS mean+16(FP), F28
	VDUP  V28.S[0], V28.S4
	VEOR  V16.B16, V16.B16, V16.B16
	CBZ   R1, norm_ssd_zero

norm_ssd_loop4:
	CMP  $4, R1
	BLT  norm_ssd_reduce

	VLD1 (R0), [V0.S4]
	VFSUB_S4(28, 0, 0)
	FCVTL_2D(0, 4)
	FCVTL2_2D(0, 5)
	VFMUL_D2(4, 4, 4)
	VFMUL_D2(5, 5, 5)
	VFADD_D2(4, 16, 16)
	VFADD_D2(5, 16, 16)

	ADD  $16, R0
	SUB  $4, R1
	B    norm_ssd_loop4

norm_ssd_reduce:
	FADDP_D(16, 0)
	CBZ  R1, norm_ssd_finalize

norm_ssd_scalar_loop:
	FMOVS (R0), F1
	FSUBS F28, F1, F1
	FCVTSD F1, F1
	FMULD F1, F1, F1
	FADDD F1, F0, F0
	ADD  $4, R0
	SUB  $1, R1
	CBNZ R1, norm_ssd_scalar_loop

norm_ssd_finalize:
	FCVTDS F0, F0
	B    norm_ssd_done

norm_ssd_zero:
	FMOVS $0, F0

norm_ssd_done:
	FMOVS F0, ret+24(FP)
	RET

// func NormApplyConstScaleBiasFloat32NEONAsm(
//     out, row *float32, count int, mean, invStdDev, scale, bias float32,
// )
TEXT ·NormApplyConstScaleBiasFloat32NEONAsm(SB), NOSPLIT, $0-52
	MOVD out+0(FP), R0
	MOVD row+8(FP), R1
	MOVD count+16(FP), R2
	FMOVS mean+24(FP), F16
	FMOVS invStdDev+28(FP), F17
	FMOVS scale+32(FP), F18
	FMOVS bias+36(FP), F19
	VDUP  V16.S[0], V16.S4
	VDUP  V17.S[0], V17.S4
	VDUP  V18.S[0], V18.S4
	VDUP  V19.S[0], V19.S4

norm_apply_loop16:
	CMP  $16, R2
	BLT  norm_apply_loop4

	VLD1 (R1), [V0.S4, V1.S4, V2.S4, V3.S4]
	VFSUB_S4(16, 0, 0)
	VFMUL_S4(17, 0, 0)
	VFMUL_S4(18, 0, 0)
	VFADD_S4(19, 0, 0)
	VFSUB_S4(16, 1, 1)
	VFMUL_S4(17, 1, 1)
	VFMUL_S4(18, 1, 1)
	VFADD_S4(19, 1, 1)
	VFSUB_S4(16, 2, 2)
	VFMUL_S4(17, 2, 2)
	VFMUL_S4(18, 2, 2)
	VFADD_S4(19, 2, 2)
	VFSUB_S4(16, 3, 3)
	VFMUL_S4(17, 3, 3)
	VFMUL_S4(18, 3, 3)
	VFADD_S4(19, 3, 3)
	VST1 [V0.S4, V1.S4, V2.S4, V3.S4], (R0)

	ADD  $64, R0
	ADD  $64, R1
	SUB  $16, R2
	B    norm_apply_loop16

norm_apply_loop4:
	CMP  $4, R2
	BLT  norm_apply_scalar_tail

	VLD1 (R1), [V0.S4]
	VFSUB_S4(16, 0, 0)
	VFMUL_S4(17, 0, 0)
	VFMUL_S4(18, 0, 0)
	VFADD_S4(19, 0, 0)
	VST1 [V0.S4], (R0)

	ADD  $16, R0
	ADD  $16, R1
	SUB  $4, R2
	B    norm_apply_loop4

norm_apply_scalar_tail:
	CBZ  R2, norm_apply_done

norm_apply_scalar_loop:
	FMOVS (R1), F0
	FSUBS F16, F0, F0
	FMULS F17, F0, F0
	FMULS F18, F0, F0
	FADDS F19, F0, F0
	FMOVS F0, (R0)
	ADD  $4, R0
	ADD  $4, R1
	SUB  $1, R2
	CBNZ R2, norm_apply_scalar_loop

norm_apply_done:
	RET
