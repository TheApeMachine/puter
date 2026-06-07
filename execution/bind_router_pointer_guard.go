package execution

import (
	"fmt"
	"unsafe"
)

func requireDispatchPointers(method string, pointers ...unsafe.Pointer) error {
	for index, pointer := range pointers {
		if pointer == nil {
			return fmt.Errorf("router %s: arg %d dispatch pointer is nil", method, index)
		}
	}

	return nil
}
