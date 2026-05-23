// Inline natural log: input in F0, result in F0 at done label. Clobbers F16-F31, R1-R9.
// Go asm FCMPD Fa, Fb sets flags for Fb - Fa.

#define AI_NEON_LOG_BF16_FE \
	FMOVD F0, F16 ;\
	FMOVD F16, R1 ;\
	LSR   $52, R1, R2 ;\
	AND   $0x7FF, R2, R2 ;\
	CMP   $0x7FF, R2 ;\
	BNE   ai_bf16_fe_log_finite ;\
	FMOVD F16, R1 ;\
	MOVD  $0x000FFFFFFFFFFFFF, R2 ;\
	AND   R2, R1, R1 ;\
	CBZ   R1, ai_bf16_fe_log_return ;\
	FMOVD F16, F0 ;\
	JMP   ai_bf16_fe_log_done ;\
ai_bf16_fe_log_finite: \
	MOVD  $0, R1 ;\
	FMOVD R1, F17 ;\
	FCMPD F17, F16 ;\
	BMI   ai_bf16_fe_log_nan ;\
	BEQ   ai_bf16_fe_log_zero ;\
	FMOVD F16, R1 ;\
	MOVD  R1, R2 ;\
	LSR   $52, R2, R2 ;\
	AND   $0x7FF, R2, R2 ;\
	SUB   $1022, R2, R8 ;\
	MOVD  $0x000FFFFFFFFFFFFF, R3 ;\
	AND   R3, R1, R1 ;\
	MOVD  $0x3FE0000000000000, R4 ;\
	ORR   R4, R1, R1 ;\
	FMOVD R1, F17 ;\
	FMOVD aiLogConsts<>+16(SB), F18 ;\
	FCMPD F18, F17 ;\
	BGE   ai_bf16_fe_log_poly ;\
	FADDD F17, F17, F17 ;\
	SUB   $1, R8, R8 ;\
ai_bf16_fe_log_poly: \
	SCVTFWD R8, F24 ;\
	MOVD  $0x3FF0000000000000, R1 ;\
	FMOVD R1, F20 ;\
	FSUBD F20, F17, F19 ;\
	MOVD  $0x4000000000000000, R1 ;\
	FMOVD R1, F20 ;\
	FADDD F20, F19, F20 ;\
	FDIVD F20, F19, F21 ;\
	FMULD F21, F21, F22 ;\
	FMULD F22, F22, F23 ;\
	FMOVD aiLogConsts<>+72(SB), F25 ;\
	FMOVD aiLogConsts<>+56(SB), F26 ;\
	FMULD F23, F25, F27 ;\
	FADDD F27, F26, F25 ;\
	FMOVD aiLogConsts<>+40(SB), F26 ;\
	FMULD F23, F25, F27 ;\
	FADDD F27, F26, F25 ;\
	FMOVD aiLogConsts<>+24(SB), F26 ;\
	FMULD F23, F25, F27 ;\
	FADDD F27, F26, F25 ;\
	FMULD F22, F25, F25 ;\
	FMOVD aiLogConsts<>+64(SB), F26 ;\
	FMOVD aiLogConsts<>+48(SB), F27 ;\
	FMULD F23, F26, F28 ;\
	FADDD F28, F27, F26 ;\
	FMOVD aiLogConsts<>+32(SB), F27 ;\
	FMULD F23, F26, F28 ;\
	FADDD F28, F27, F26 ;\
	FMULD F23, F26, F26 ;\
	FADDD F25, F26, F29 ;\
	FMULD F19, F19, F28 ;\
	MOVD  $0x3FE0000000000000, R1 ;\
	FMOVD R1, F20 ;\
	FMULD F28, F20, F28 ;\
	FADDD F29, F28, F30 ;\
	FMULD F21, F30, F30 ;\
	FMOVD aiLogConsts<>+8(SB), F31 ;\
	FMULD F24, F31, F31 ;\
	FADDD F31, F30, F30 ;\
	FSUBD F30, F28, F28 ;\
	FSUBD F19, F28, F28 ;\
	FMOVD aiLogConsts<>+0(SB), F26 ;\
	FMULD F24, F26, F26 ;\
	FSUBD F28, F26, F0 ;\
	JMP   ai_bf16_fe_log_done ;\
ai_bf16_fe_log_zero: \
	FMOVD aiLogConsts<>+80(SB), F0 ;\
	JMP   ai_bf16_fe_log_done ;\
ai_bf16_fe_log_nan: \
	FMOVD aiLogConsts<>+88(SB), F0 ;\
	JMP   ai_bf16_fe_log_done ;\
ai_bf16_fe_log_return: \
	FMOVD F16, F0 ;\
ai_bf16_fe_log_done:

#define AI_NEON_LOG_BF16_EFE \
	FMOVD F0, F16 ;\
	FMOVD F16, R1 ;\
	LSR   $52, R1, R2 ;\
	AND   $0x7FF, R2, R2 ;\
	CMP   $0x7FF, R2 ;\
	BNE   ai_bf16_efe_log_finite ;\
	FMOVD F16, R1 ;\
	MOVD  $0x000FFFFFFFFFFFFF, R2 ;\
	AND   R2, R1, R1 ;\
	CBZ   R1, ai_bf16_efe_log_return ;\
	FMOVD F16, F0 ;\
	JMP   ai_bf16_efe_log_done ;\
ai_bf16_efe_log_finite: \
	MOVD  $0, R1 ;\
	FMOVD R1, F17 ;\
	FCMPD F17, F16 ;\
	BMI   ai_bf16_efe_log_nan ;\
	BEQ   ai_bf16_efe_log_zero ;\
	FMOVD F16, R1 ;\
	MOVD  R1, R2 ;\
	LSR   $52, R2, R2 ;\
	AND   $0x7FF, R2, R2 ;\
	SUB   $1022, R2, R8 ;\
	MOVD  $0x000FFFFFFFFFFFFF, R3 ;\
	AND   R3, R1, R1 ;\
	MOVD  $0x3FE0000000000000, R4 ;\
	ORR   R4, R1, R1 ;\
	FMOVD R1, F17 ;\
	FMOVD aiLogConsts<>+16(SB), F18 ;\
	FCMPD F18, F17 ;\
	BGE   ai_bf16_efe_log_poly ;\
	FADDD F17, F17, F17 ;\
	SUB   $1, R8, R8 ;\
ai_bf16_efe_log_poly: \
	SCVTFWD R8, F24 ;\
	MOVD  $0x3FF0000000000000, R1 ;\
	FMOVD R1, F20 ;\
	FSUBD F20, F17, F19 ;\
	MOVD  $0x4000000000000000, R1 ;\
	FMOVD R1, F20 ;\
	FADDD F20, F19, F20 ;\
	FDIVD F20, F19, F21 ;\
	FMULD F21, F21, F22 ;\
	FMULD F22, F22, F23 ;\
	FMOVD aiLogConsts<>+72(SB), F25 ;\
	FMOVD aiLogConsts<>+56(SB), F26 ;\
	FMULD F23, F25, F27 ;\
	FADDD F27, F26, F25 ;\
	FMOVD aiLogConsts<>+40(SB), F26 ;\
	FMULD F23, F25, F27 ;\
	FADDD F27, F26, F25 ;\
	FMOVD aiLogConsts<>+24(SB), F26 ;\
	FMULD F23, F25, F27 ;\
	FADDD F27, F26, F25 ;\
	FMULD F22, F25, F25 ;\
	FMOVD aiLogConsts<>+64(SB), F26 ;\
	FMOVD aiLogConsts<>+48(SB), F27 ;\
	FMULD F23, F26, F28 ;\
	FADDD F28, F27, F26 ;\
	FMOVD aiLogConsts<>+32(SB), F27 ;\
	FMULD F23, F26, F28 ;\
	FADDD F28, F27, F26 ;\
	FMULD F23, F26, F26 ;\
	FADDD F25, F26, F29 ;\
	FMULD F19, F19, F28 ;\
	MOVD  $0x3FE0000000000000, R1 ;\
	FMOVD R1, F20 ;\
	FMULD F28, F20, F28 ;\
	FADDD F29, F28, F30 ;\
	FMULD F21, F30, F30 ;\
	FMOVD aiLogConsts<>+8(SB), F31 ;\
	FMULD F24, F31, F31 ;\
	FADDD F31, F30, F30 ;\
	FSUBD F30, F28, F28 ;\
	FSUBD F19, F28, F28 ;\
	FMOVD aiLogConsts<>+0(SB), F26 ;\
	FMULD F24, F26, F26 ;\
	FSUBD F28, F26, F0 ;\
	JMP   ai_bf16_efe_log_done ;\
ai_bf16_efe_log_zero: \
	FMOVD aiLogConsts<>+80(SB), F0 ;\
	JMP   ai_bf16_efe_log_done ;\
ai_bf16_efe_log_nan: \
	FMOVD aiLogConsts<>+88(SB), F0 ;\
	JMP   ai_bf16_efe_log_done ;\
ai_bf16_efe_log_return: \
	FMOVD F16, F0 ;\
ai_bf16_efe_log_done:

#define AI_NEON_LOG_FP16_FE \
	FMOVD F0, F16 ;\
	FMOVD F16, R1 ;\
	LSR   $52, R1, R2 ;\
	AND   $0x7FF, R2, R2 ;\
	CMP   $0x7FF, R2 ;\
	BNE   ai_fp16_fe_log_finite ;\
	FMOVD F16, R1 ;\
	MOVD  $0x000FFFFFFFFFFFFF, R2 ;\
	AND   R2, R1, R1 ;\
	CBZ   R1, ai_fp16_fe_log_return ;\
	FMOVD F16, F0 ;\
	JMP   ai_fp16_fe_log_done ;\
ai_fp16_fe_log_finite: \
	MOVD  $0, R1 ;\
	FMOVD R1, F17 ;\
	FCMPD F17, F16 ;\
	BMI   ai_fp16_fe_log_nan ;\
	BEQ   ai_fp16_fe_log_zero ;\
	FMOVD F16, R1 ;\
	MOVD  R1, R2 ;\
	LSR   $52, R2, R2 ;\
	AND   $0x7FF, R2, R2 ;\
	SUB   $1022, R2, R8 ;\
	MOVD  $0x000FFFFFFFFFFFFF, R3 ;\
	AND   R3, R1, R1 ;\
	MOVD  $0x3FE0000000000000, R4 ;\
	ORR   R4, R1, R1 ;\
	FMOVD R1, F17 ;\
	FMOVD aiLogConsts<>+16(SB), F18 ;\
	FCMPD F18, F17 ;\
	BGE   ai_fp16_fe_log_poly ;\
	FADDD F17, F17, F17 ;\
	SUB   $1, R8, R8 ;\
ai_fp16_fe_log_poly: \
	SCVTFWD R8, F24 ;\
	MOVD  $0x3FF0000000000000, R1 ;\
	FMOVD R1, F20 ;\
	FSUBD F20, F17, F19 ;\
	MOVD  $0x4000000000000000, R1 ;\
	FMOVD R1, F20 ;\
	FADDD F20, F19, F20 ;\
	FDIVD F20, F19, F21 ;\
	FMULD F21, F21, F22 ;\
	FMULD F22, F22, F23 ;\
	FMOVD aiLogConsts<>+72(SB), F25 ;\
	FMOVD aiLogConsts<>+56(SB), F26 ;\
	FMULD F23, F25, F27 ;\
	FADDD F27, F26, F25 ;\
	FMOVD aiLogConsts<>+40(SB), F26 ;\
	FMULD F23, F25, F27 ;\
	FADDD F27, F26, F25 ;\
	FMOVD aiLogConsts<>+24(SB), F26 ;\
	FMULD F23, F25, F27 ;\
	FADDD F27, F26, F25 ;\
	FMULD F22, F25, F25 ;\
	FMOVD aiLogConsts<>+64(SB), F26 ;\
	FMOVD aiLogConsts<>+48(SB), F27 ;\
	FMULD F23, F26, F28 ;\
	FADDD F28, F27, F26 ;\
	FMOVD aiLogConsts<>+32(SB), F27 ;\
	FMULD F23, F26, F28 ;\
	FADDD F28, F27, F26 ;\
	FMULD F23, F26, F26 ;\
	FADDD F25, F26, F29 ;\
	FMULD F19, F19, F28 ;\
	MOVD  $0x3FE0000000000000, R1 ;\
	FMOVD R1, F20 ;\
	FMULD F28, F20, F28 ;\
	FADDD F29, F28, F30 ;\
	FMULD F21, F30, F30 ;\
	FMOVD aiLogConsts<>+8(SB), F31 ;\
	FMULD F24, F31, F31 ;\
	FADDD F31, F30, F30 ;\
	FSUBD F30, F28, F28 ;\
	FSUBD F19, F28, F28 ;\
	FMOVD aiLogConsts<>+0(SB), F26 ;\
	FMULD F24, F26, F26 ;\
	FSUBD F28, F26, F0 ;\
	JMP   ai_fp16_fe_log_done ;\
ai_fp16_fe_log_zero: \
	FMOVD aiLogConsts<>+80(SB), F0 ;\
	JMP   ai_fp16_fe_log_done ;\
ai_fp16_fe_log_nan: \
	FMOVD aiLogConsts<>+88(SB), F0 ;\
	JMP   ai_fp16_fe_log_done ;\
ai_fp16_fe_log_return: \
	FMOVD F16, F0 ;\
ai_fp16_fe_log_done:

#define AI_NEON_LOG_FP16_EFE \
	FMOVD F0, F16 ;\
	FMOVD F16, R1 ;\
	LSR   $52, R1, R2 ;\
	AND   $0x7FF, R2, R2 ;\
	CMP   $0x7FF, R2 ;\
	BNE   ai_fp16_efe_log_finite ;\
	FMOVD F16, R1 ;\
	MOVD  $0x000FFFFFFFFFFFFF, R2 ;\
	AND   R2, R1, R1 ;\
	CBZ   R1, ai_fp16_efe_log_return ;\
	FMOVD F16, F0 ;\
	JMP   ai_fp16_efe_log_done ;\
ai_fp16_efe_log_finite: \
	MOVD  $0, R1 ;\
	FMOVD R1, F17 ;\
	FCMPD F17, F16 ;\
	BMI   ai_fp16_efe_log_nan ;\
	BEQ   ai_fp16_efe_log_zero ;\
	FMOVD F16, R1 ;\
	MOVD  R1, R2 ;\
	LSR   $52, R2, R2 ;\
	AND   $0x7FF, R2, R2 ;\
	SUB   $1022, R2, R8 ;\
	MOVD  $0x000FFFFFFFFFFFFF, R3 ;\
	AND   R3, R1, R1 ;\
	MOVD  $0x3FE0000000000000, R4 ;\
	ORR   R4, R1, R1 ;\
	FMOVD R1, F17 ;\
	FMOVD aiLogConsts<>+16(SB), F18 ;\
	FCMPD F18, F17 ;\
	BGE   ai_fp16_efe_log_poly ;\
	FADDD F17, F17, F17 ;\
	SUB   $1, R8, R8 ;\
ai_fp16_efe_log_poly: \
	SCVTFWD R8, F24 ;\
	MOVD  $0x3FF0000000000000, R1 ;\
	FMOVD R1, F20 ;\
	FSUBD F20, F17, F19 ;\
	MOVD  $0x4000000000000000, R1 ;\
	FMOVD R1, F20 ;\
	FADDD F20, F19, F20 ;\
	FDIVD F20, F19, F21 ;\
	FMULD F21, F21, F22 ;\
	FMULD F22, F22, F23 ;\
	FMOVD aiLogConsts<>+72(SB), F25 ;\
	FMOVD aiLogConsts<>+56(SB), F26 ;\
	FMULD F23, F25, F27 ;\
	FADDD F27, F26, F25 ;\
	FMOVD aiLogConsts<>+40(SB), F26 ;\
	FMULD F23, F25, F27 ;\
	FADDD F27, F26, F25 ;\
	FMOVD aiLogConsts<>+24(SB), F26 ;\
	FMULD F23, F25, F27 ;\
	FADDD F27, F26, F25 ;\
	FMULD F22, F25, F25 ;\
	FMOVD aiLogConsts<>+64(SB), F26 ;\
	FMOVD aiLogConsts<>+48(SB), F27 ;\
	FMULD F23, F26, F28 ;\
	FADDD F28, F27, F26 ;\
	FMOVD aiLogConsts<>+32(SB), F27 ;\
	FMULD F23, F26, F28 ;\
	FADDD F28, F27, F26 ;\
	FMULD F23, F26, F26 ;\
	FADDD F25, F26, F29 ;\
	FMULD F19, F19, F28 ;\
	MOVD  $0x3FE0000000000000, R1 ;\
	FMOVD R1, F20 ;\
	FMULD F28, F20, F28 ;\
	FADDD F29, F28, F30 ;\
	FMULD F21, F30, F30 ;\
	FMOVD aiLogConsts<>+8(SB), F31 ;\
	FMULD F24, F31, F31 ;\
	FADDD F31, F30, F30 ;\
	FSUBD F30, F28, F28 ;\
	FSUBD F19, F28, F28 ;\
	FMOVD aiLogConsts<>+0(SB), F26 ;\
	FMULD F24, F26, F26 ;\
	FSUBD F28, F26, F0 ;\
	JMP   ai_fp16_efe_log_done ;\
ai_fp16_efe_log_zero: \
	FMOVD aiLogConsts<>+80(SB), F0 ;\
	JMP   ai_fp16_efe_log_done ;\
ai_fp16_efe_log_nan: \
	FMOVD aiLogConsts<>+88(SB), F0 ;\
	JMP   ai_fp16_efe_log_done ;\
ai_fp16_efe_log_return: \
	FMOVD F16, F0 ;\
ai_fp16_efe_log_done:

#define AI_F64_LOG_BF16_FE(in, out) \
	FMOVD in, F0 ;\
	AI_NEON_LOG_BF16_FE ;\
	FMOVD F0, out

#define AI_F64_LOG_BF16_EFE(in, out) \
	FMOVD in, F0 ;\
	AI_NEON_LOG_BF16_EFE ;\
	FMOVD F0, out

#define AI_F64_LOG_FP16_FE(in, out) \
	FMOVD in, F0 ;\
	AI_NEON_LOG_FP16_FE ;\
	FMOVD F0, out

#define AI_F64_LOG_FP16_EFE(in, out) \
	FMOVD in, F0 ;\
	AI_NEON_LOG_FP16_EFE ;\
	FMOVD F0, out
