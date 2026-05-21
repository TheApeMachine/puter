//go:build darwin && cgo

package metal

func pospopCount8Generic(counts *[8]int, buf []uint8) {
	for index := range buf {
		value := buf[index]

		for bit := range 8 {
			counts[bit] += int(value >> bit & 1)
		}
	}
}

func pospopCount16Generic(counts *[16]int, buf []uint16) {
	for index := range buf {
		value := buf[index]

		for bit := range 16 {
			counts[bit] += int(value >> bit & 1)
		}
	}
}

func pospopCount32Generic(counts *[32]int, buf []uint32) {
	for index := range buf {
		value := buf[index]

		for bit := range 32 {
			counts[bit] += int(value >> bit & 1)
		}
	}
}

func pospopCount64Generic(counts *[64]int, buf []uint64) {
	for index := range buf {
		value := buf[index]

		for bit := range 64 {
			counts[bit] += int(value >> bit & 1)
		}
	}
}
