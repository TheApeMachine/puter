// SPDX-License-Identifier: Apache-2.0
// Scalar float64 natural log matching math/log.go for typed active inference.
#include "textflag.h"
#include "log_f64_scalar_neon.inc"

DATA aiLogConsts<>+0(SB)/8, $0x3FE62E42FEE00000   // ln2Hi
DATA aiLogConsts<>+8(SB)/8, $0x3DEA39EF35793C76   // ln2Lo
DATA aiLogConsts<>+16(SB)/8, $0x3FE6A09E667F3BCD  // sqrt(2)/2
DATA aiLogConsts<>+24(SB)/8, $0x3FE5555555555593  // L1
DATA aiLogConsts<>+32(SB)/8, $0x3FD999999997FA04  // L2
DATA aiLogConsts<>+40(SB)/8, $0x3FD2492494229359  // L3
DATA aiLogConsts<>+48(SB)/8, $0x3FCC71C51D8E78AF  // L4
DATA aiLogConsts<>+56(SB)/8, $0x3FC7466496CB03DE  // L5
DATA aiLogConsts<>+64(SB)/8, $0x3FC39A09D078C69F  // L6
DATA aiLogConsts<>+72(SB)/8, $0x3FC2F112DF3E5244  // L7
DATA aiLogConsts<>+80(SB)/8, $0xFFF0000000000000  // -Inf
DATA aiLogConsts<>+88(SB)/8, $0x7FF8000000000000  // quiet NaN
GLOBL aiLogConsts<>(SB), RODATA|NOPTR, $96

// aiNeonLogF64 computes ln(x) in F0 using F0 as input. Clobbers F16-F31, R1-R9.
TEXT aiNeonLogF64(SB), NOSPLIT, $0-0
	AI_NEON_LOG_BF16_FE
	RET

// func activeInferenceLogF64NEONAsm(value float64) float64
TEXT ·activeInferenceLogF64NEONAsm(SB), NOSPLIT, $16-16
	FMOVD value+0(FP), F0
	CALL  aiNeonLogF64(SB)
	FMOVD F0, ret+8(FP)
	RET
