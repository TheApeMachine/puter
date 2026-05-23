package metal

/*
CountString is host-side PosPop on CPU today; Metal does not implement PosPop.
*/
func (backend *Backend) CountString(counts *[8]int, str string) {
	panic("metal: pospop not implemented")
}

func (backend *Backend) Count8(counts *[8]int, buf []uint8) {
	panic("metal: pospop not implemented")
}

func (backend *Backend) Count16(counts *[16]int, buf []uint16) {
	panic("metal: pospop not implemented")
}

func (backend *Backend) Count32(counts *[32]int, buf []uint32) {
	panic("metal: pospop not implemented")
}

func (backend *Backend) Count64(counts *[64]int, buf []uint64) {
	panic("metal: pospop not implemented")
}
