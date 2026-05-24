package pospop

var defaultPosPop = New()

func Count16(counts *[16]int, buf []uint16) {
	defaultPosPop.Count16(counts, buf)
}

func Count32(counts *[32]int, buf []uint32) {
	defaultPosPop.Count32(counts, buf)
}

func Count64(counts *[64]int, buf []uint64) {
	defaultPosPop.Count64(counts, buf)
}

func Count8(counts *[8]int, buf []uint8) {
	defaultPosPop.Count8(counts, buf)
}

func CountString(counts *[8]int, str string) {
	defaultPosPop.CountString(counts, str)
}
