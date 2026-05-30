#include "textflag.h"
TEXT ·probeUnused(SB), NOSPLIT, $0-0
	VPBROADCASTW (AX), X2
	VCVTPH2PS X2, Y4
	VMULPS Y4, Y4, Y4
	RET
