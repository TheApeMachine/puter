// SPDX-License-Identifier: Apache-2.0
// bfloat16 typed FreeEnergy / ExpectedFreeEnergy: load bf16 → f64, inline log, f64 accumulate, bf16 store.
#include "textflag.h"
#include "log_f64_scalar_neon.inc"

#define AI_BF16_LOAD_F64(ptr, fd) \
	MOVHU (ptr), R6 ;\
	LSL  $16, R6, R6 ;\
	FMOVS R6, F1 ;\
	FCVTSD F1, fd

#define AI_F64_CLAMP_POS(fd) \
	MOVD  $0x3F506024DD2F1AA0, R6 ;\
	FMOVD R6, F2 ;\
	FMAXD fd, F2, fd

#define AI_BF16_STORE_FE_SUM(fd) \
	FCVTSD fd, F0 ;\
	FMOVS F0, R6 ;\
	LSR  $16, R6, R6 ;\
	MOVH R6, ret+32(FP)

#define AI_BF16_STORE_EFE_SUM(fd) \
	FCVTSD fd, F0 ;\
	FMOVS F0, R6 ;\
	LSR  $16, R6, R6 ;\
	MOVH R6, ret+40(FP)

// func freeEnergyBFloat16NEONBridge(likelihood, posterior, prior uintptr, count int) uint16
TEXT ·freeEnergyBFloat16NEONBridge(SB), NOSPLIT, $0-34
	MOVD likelihood+0(FP), R10
	MOVD posterior+8(FP), R11
	MOVD prior+16(FP), R12
	MOVD count+24(FP), R3
	FMOVD $0, F14
	FMOVD $0, F15
	CBZ  R3, ai_bf16_fe_f64_store

ai_bf16_fe_f64_loop:
	AI_BF16_LOAD_F64(R10, F8)
	AI_BF16_LOAD_F64(R11, F9)
	AI_BF16_LOAD_F64(R12, F10)
	FMOVD F8, F12
	FMOVD F9, F11
	FMOVD F10, F13
	AI_F64_CLAMP_POS(F12)
	AI_F64_CLAMP_POS(F11)
	AI_F64_CLAMP_POS(F13)
	AI_F64_LOG_BF16_FE(F12, F4)
	AI_F64_LOG_BF16_FE(F11, F5)
	AI_F64_LOG_BF16_FE(F13, F6)
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
	CBNZ R3, ai_bf16_fe_f64_loop

ai_bf16_fe_f64_store:
	FADDD F15, F14, F14
	AI_BF16_STORE_FE_SUM(F14)
	RET

// func expectedFreeEnergyBFloat16NEONBridge(
//     predictedObs, preferredObs, predictedState uintptr,
//     obsCount, stateCount int,
// ) uint16
TEXT ·expectedFreeEnergyBFloat16NEONBridge(SB), NOSPLIT, $0-42
	MOVD predictedObs+0(FP), R0
	MOVD preferredObs+8(FP), R1
	MOVD predictedState+16(FP), R2
	MOVD obsCount+24(FP), R3
	MOVD stateCount+32(FP), R4
	FMOVD $0, F14
	CBZ  R3, ai_bf16_efe_f64_state

ai_bf16_efe_f64_obs_loop:
	AI_BF16_LOAD_F64(R0, F8)
	AI_BF16_LOAD_F64(R1, F9)
	FMOVD F8, F12
	FMOVD F9, F13
	AI_F64_CLAMP_POS(F12)
	AI_F64_CLAMP_POS(F13)
	AI_F64_LOG_BF16_EFE(F12, F4)
	AI_F64_LOG_BF16_EFE(F13, F5)
	FSUBD F5, F4, F4
	FMULD F8, F4, F4
	FADDD F4, F14, F14
	ADD  $2, R0
	ADD  $2, R1
	SUB  $1, R3
	CBNZ R3, ai_bf16_efe_f64_obs_loop

ai_bf16_efe_f64_state:
	MOVD predictedState+16(FP), R2
	MOVD stateCount+32(FP), R4
	CBZ  R4, ai_bf16_efe_f64_store

ai_bf16_efe_f64_state_loop:
	AI_BF16_LOAD_F64(R2, F8)
	FMOVD F8, F9
	FMOVD F8, F12
	AI_F64_CLAMP_POS(F12)
	AI_F64_LOG_BF16_EFE(F12, F4)
	FNEGD F4, F4
	FMULD F9, F4, F4
	FADDD F4, F14, F14
	ADD  $2, R2
	SUB  $1, R4
	CBNZ R4, ai_bf16_efe_f64_state_loop

ai_bf16_efe_f64_store:
	AI_BF16_STORE_EFE_SUM(F14)
	RET
