package cuda

import (
	"github.com/theapemachine/puter/device/cpu/pospop"
)

func (backend *Backend) CountString(counts *[8]int, str string) {
	pospop.CountString(counts, str)
}

func (backend *Backend) Count8(counts *[8]int, buf []uint8) {
	pospop.Count8(counts, buf)
}

func (backend *Backend) Count16(counts *[16]int, buf []uint16) {
	pospop.Count16(counts, buf)
}

func (backend *Backend) Count32(counts *[32]int, buf []uint32) {
	pospop.Count32(counts, buf)
}

func (backend *Backend) Count64(counts *[64]int, buf []uint64) {
	pospop.Count64(counts, buf)
}
