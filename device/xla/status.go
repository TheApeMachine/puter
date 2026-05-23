package xla

import "fmt"

/*
XLAStatus mirrors the C bridge status payload.
*/
type XLAStatus struct {
	Code    int
	Message string
}

func statusError(status XLAStatus) error {
	if status.Code == 0 {
		return nil
	}

	if status.Message == "" {
		return fmt.Errorf("xla bridge error %d", status.Code)
	}

	return fmt.Errorf("xla bridge: %s", status.Message)
}
