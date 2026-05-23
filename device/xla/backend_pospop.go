package xla

/*
PosPop is host-side preprocessing on CPU. XLA satisfies device.Backend by
rejecting PosPop entry points at runtime.
*/
func (backend *Backend) CountString(counts *[8]int, str string) {
	panic("xla: pospop not implemented")
}

func (backend *Backend) Count8(counts *[8]int, buf []uint8) {
	panic("xla: pospop not implemented")
}

func (backend *Backend) Count16(counts *[16]int, buf []uint16) {
	panic("xla: pospop not implemented")
}

func (backend *Backend) Count32(counts *[32]int, buf []uint32) {
	panic("xla: pospop not implemented")
}

func (backend *Backend) Count64(counts *[64]int, buf []uint64) {
	panic("xla: pospop not implemented")
}
