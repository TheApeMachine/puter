// SPDX-License-Identifier: Apache-2.0
// float16 typed FreeEnergy / ExpectedFreeEnergy: load fp16 → f64, log, f64 accumulate, fp16 store.
#include "textflag.h"

#define VFCVTL_4S(n, d) WORD $(0x0E217800 | ((n) << 5) | (d))
#define VFADD_S4(m, n, d)   WORD $(0x4E20D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFSUB_S4(m, n, d)   WORD $(0x4EA0D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_S4(m, n, d)   WORD $(0x6E20DC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFDIV_S4(m, n, d)   WORD $(0x6E20FC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFMLA_S4(m, n, d)   WORD $(0x4E20CC00 | ((m) << 16) | ((n) << 5) | (d))
#define VUSHR_S4_BY23(n, d) WORD $(0x6F290400 | ((n) << 5) | (d))
#define VISUB_S4(m, n, d)   WORD $(0x6EA08400 | ((m) << 16) | ((n) << 5) | (d))
#define VAND_B16(m, n, d)   WORD $(0x4E201C00 | ((m) << 16) | ((n) << 5) | (d))
#define VORR_B16(m, n, d)   WORD $(0x4EA01C00 | ((m) << 16) | ((n) << 5) | (d))
#define VSCVTF_S4(n, d)     WORD $(0x4E21D800 | ((n) << 5) | (d))
#define VMAX_S4(m, n, d)    WORD $(0x4E20F400 | ((m) << 16) | ((n) << 5) | (d))
#define VMOV_B16(src, dst)  WORD $(0x4EA01C00 | ((src) << 16) | ((src) << 5) | (dst))

#define AI_TYPED_LOAD_LOG_MASKS \
	MOVD $0x007FFFFF, R6 ;\
	VMOV R6, V24.S[0] ;\
	VDUP V24.S[0], V24.S4 ;\
	MOVD $0x3F800000, R6 ;\
	VMOV R6, V25.S[0] ;\
	VDUP V25.S[0], V25.S4 ;\
	MOVD $127, R6 ;\
	VMOV R6, V26.S[0] ;\
	VDUP V26.S[0], V26.S4

#define AI_TYPED_RELOAD_LOG_POLY \
	MOVD $aiTypedLogC<>(SB), R13 ;\
	FMOVS  0(R13), F16 ;\
	VDUP V16.S[0], V16.S4 ;\
	FMOVS  4(R13), F17 ;\
	VDUP V17.S[0], V17.S4 ;\
	FMOVS  8(R13), F18 ;\
	VDUP V18.S[0], V18.S4 ;\
	FMOVS 12(R13), F19 ;\
	VDUP V19.S[0], V19.S4 ;\
	FMOVS 16(R13), F20 ;\
	VDUP V20.S[0], V20.S4 ;\
	FMOVS 20(R13), F21 ;\
	VDUP V21.S[0], V21.S4 ;\
	FMOVS 24(R13), F22 ;\
	VDUP V22.S[0], V22.S4 ;\
	FMOVS 28(R13), F23 ;\
	VDUP V23.S[0], V23.S4

#define AI_TYPED_LOAD_LOG_CONSTS \
	AI_TYPED_RELOAD_LOG_POLY ;\
	AI_TYPED_LOAD_LOG_MASKS

#define AI_TYPED_NEON_LOG4(in, out) \
	VUSHR_S4_BY23(in, 1) ;\
	VISUB_S4(26, 1, 1) ;\
	VAND_B16(24, in, 2) ;\
	VORR_B16(25, 2, 2) ;\
	VSCVTF_S4(1, 1) ;\
	VFSUB_S4(17, 2, 3) ;\
	VFADD_S4(17, 2, 4) ;\
	VFDIV_S4(4, 3, 5) ;\
	VFMUL_S4(5, 5, 6) ;\
	VMOV_B16(18, 7) ;\
	VMOV_B16(19, 8) ; VFMLA_S4(6, 7, 8) ;\
	VMOV_B16(20, 7) ; VFMLA_S4(6, 8, 7) ;\
	VMOV_B16(21, 8) ; VFMLA_S4(6, 7, 8) ;\
	VMOV_B16(22, 7) ; VFMLA_S4(6, 8, 7) ;\
	VMOV_B16(17, 8) ; VFMLA_S4(6, 7, 8) ;\
	VFMUL_S4(5, 8, 8) ;\
	VFMUL_S4(23, 8, 8) ;\
	VFMLA_S4(16, 1, 8) ;\
	VMOV_B16(8, out)

#define AI_TYPED_INIT_LOG \
	MOVD $0x2b8cbccc, R17 ;\
	VMOV R17, V31.S[0] ;\
	VDUP V31.S[0], V31.S4 ;\
	AI_TYPED_LOAD_LOG_CONSTS

#define AI_FP16_LOAD_F64(ptr, fd) \
	MOVHU (ptr), R6 ;\
	VMOV R6, V0.H[0] ;\
	VFCVTL_4S(0, 0) ;\
	FMOVS F0, R6 ;\
	FMOVS R6, F0 ;\
	FCVTSD F0, fd

#define AI_F64_CLAMP_POS(fd) \
	FMOVD aiEpsF64<>(SB), F2 ;\
	FMAXD fd, F2, fd

#define AI_F64_LOG_TO(in, out) \
	FCVTSD in, F1 ;\
	FMOVS F1, F3 ;\
	VDUP V3.S[0], V3.S4 ;\
	VMAX_S4(31, 3, 3) ;\
	AI_TYPED_NEON_LOG4(3, 10) ;\
	FMOVS F10, F1 ;\
	FCVTDS F1, out

#define AI_FP16_STORE_FE_SUM(fd) \
	FCVTSD fd, F0 ;\
	FCVTSH F0, F0 ;\
	FMOVD F0, R6 ;\
	MOVH R6, ret+32(FP)

#define AI_FP16_STORE_EFE_SUM(fd) \
	FCVTSD fd, F0 ;\
	FCVTSH F0, F0 ;\
	FMOVD F0, R6 ;\
	MOVH R6, ret+40(FP)

TEXT ·FreeEnergyFloat16NEONAsm(SB), NOSPLIT, $0-34
	MOVD likelihood+0(FP), R10
	MOVD posterior+8(FP), R11
	MOVD prior+16(FP), R12
	MOVD count+24(FP), R3
	AI_TYPED_INIT_LOG
	FMOVD $0, F14
	FMOVD $0, F15
	CBZ  R3, ai_fp16_fe_f64_store

ai_fp16_fe_f64_loop:
	AI_FP16_LOAD_F64(R10, F8)
	AI_FP16_LOAD_F64(R11, F9)
	AI_FP16_LOAD_F64(R12, F10)
	FMOVD F8, F12
	FMOVD F9, F11
	FMOVD F10, F13
	AI_F64_CLAMP_POS(F12)
	AI_F64_CLAMP_POS(F11)
	AI_F64_CLAMP_POS(F13)
	AI_F64_LOG_TO(F12, F4)
	AI_F64_LOG_TO(F11, F5)
	AI_F64_LOG_TO(F13, F6)
	FNEGD F4, F4
	FMULD F9, F4, F4
	FADDD F4, F14, F14
	FSUBD F6, F5, F5
	FMULD F9, F5, F5
	FADDD F5, F15, F15
	ADD  $2, R10
	ADD  $2, R11
	ADD  $2, R12
	SUB  $1, R3
	CBNZ R3, ai_fp16_fe_f64_loop

ai_fp16_fe_f64_store:
	FADDD F15, F14, F14
	AI_FP16_STORE_FE_SUM(F14)
	RET

TEXT ·ExpectedFreeEnergyFloat16NEONAsm(SB), NOSPLIT, $0-42
	MOVD predictedObs+0(FP), R0
	MOVD preferredObs+8(FP), R1
	MOVD predictedState+16(FP), R2
	MOVD obsCount+24(FP), R3
	MOVD stateCount+32(FP), R4
	AI_TYPED_INIT_LOG
	FMOVD $0, F14

ai_fp16_efe_f64_obs:
	CBZ  R3, ai_fp16_efe_f64_state

ai_fp16_efe_f64_obs_loop:
	AI_FP16_LOAD_F64(R0, F8)
	AI_FP16_LOAD_F64(R1, F9)
	FMOVD F8, F12
	FMOVD F9, F13
	AI_F64_CLAMP_POS(F12)
	AI_F64_CLAMP_POS(F13)
	AI_F64_LOG_TO(F12, F4)
	AI_F64_LOG_TO(F13, F5)
	FSUBD F5, F4, F4
	FMULD F8, F4, F4
	FADDD F4, F14, F14
	ADD  $2, R0
	ADD  $2, R1
	SUB  $1, R3
	CBNZ R3, ai_fp16_efe_f64_obs_loop
	B    ai_fp16_efe_f64_state

ai_fp16_efe_f64_state:
	MOVD predictedState+16(FP), R2
	MOVD stateCount+32(FP), R4
	CBZ  R4, ai_fp16_efe_f64_store

ai_fp16_efe_f64_state_loop:
	AI_FP16_LOAD_F64(R2, F8)
	FMOVD F8, F9
	FMOVD F8, F12
	AI_F64_CLAMP_POS(F12)
	AI_F64_LOG_TO(F12, F4)
	FNEGD F4, F4
	FMULD F9, F4, F4
	FADDD F4, F14, F14
	ADD  $2, R2
	SUB  $1, R4
	CBNZ R4, ai_fp16_efe_f64_state_loop

ai_fp16_efe_f64_store:
	AI_FP16_STORE_EFE_SUM(F14)
	RET
