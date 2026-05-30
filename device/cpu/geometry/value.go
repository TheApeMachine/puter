package geometry

/*
ValueToken is the 512-bit structural word band used by PhaseDial encoding.
Eight uint64 lanes mirror the canonical properties region layout.
*/
type ValueToken [8]uint64

/*
ValueTokenFromBytes packs raw bytes into the first eight uint64 lanes.
Bytes beyond 64 are ignored.
*/
func ValueTokenFromBytes(payload []byte) ValueToken {
	var token ValueToken

	for byteIndex, payloadByte := range payload {
		if byteIndex >= 64 {
			break
		}

		wordIndex := byteIndex / 8
		shift := uint((byteIndex % 8) * 8)
		token[wordIndex] |= uint64(payloadByte) << shift
	}

	return token
}

/*
Word returns one uint64 lane from the token.
*/
func (token *ValueToken) Word(wordIndex int) uint64 {
	if wordIndex < 0 || wordIndex >= len(token) {
		return 0
	}

	return token[wordIndex]
}
