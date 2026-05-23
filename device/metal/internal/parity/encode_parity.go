package parity

/*
AssertEncodedSlicesEqual fails when reduced-precision storage bytes differ.
*/
func AssertEncodedSlicesEqual(
	testingTB interface {
		Helper()
		Fatal(args ...any)
	},
	got, want []byte,
) {
	testingTB.Helper()

	if len(got) != len(want) {
		testingTB.Fatal("byte length mismatch got=", len(got), " want=", len(want))
	}

	for index := range got {
		if got[index] == want[index] {
			continue
		}

		testingTB.Fatal(
			"storage byte ", index,
			" got=", got[index],
			" want=", want[index],
		)
	}
}
