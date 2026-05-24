#include "textflag.h"
TEXT ·TestFMADDS(SB), NOSPLIT, $0-0
    FMOVS $1.0, F0
    FMOVS $2.0, F1
    FMOVS $3.0, F2
    FMADDS F3, F0, F1, F2
    RET
