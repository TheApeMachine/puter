//go:build darwin && cgo

package metal

import "unsafe"

func (backend *Backend) CountString(counts *[8]int, str string) {
	buf := unsafe.Slice(unsafe.StringData(str), len(str))
	backend.Count8(counts, buf)
}

func (backend *Backend) Count8(counts *[8]int, buf []uint8) {
	pospopCount8Generic(counts, buf)
}

func (backend *Backend) Count16(counts *[16]int, buf []uint16) {
	pospopCount16Generic(counts, buf)
}

func (backend *Backend) Count32(counts *[32]int, buf []uint32) {
	pospopCount32Generic(counts, buf)
}

func (backend *Backend) Count64(counts *[64]int, buf []uint64) {
	pospopCount64Generic(counts, buf)
}
