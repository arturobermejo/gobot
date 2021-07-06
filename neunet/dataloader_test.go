package neunet

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestGetVocab(t *testing.T) {
	vocab := GetVocab([]string{"one two", "three four"})
	expected := map[string]int{"one": 0, "two": 1, "three": 2, "four": 3}

	assert.Equal(t, expected, vocab)
}

func TestOneHotEncode(t *testing.T) {
	vocab := map[string]int{"one": 0, "two": 1, "three": 2, "four": 3}
	r := EncodeInputs([]string{"two one one", "one any four"}, vocab, 0)
	expected := mat.NewDense(2, 4, []float64{1, 1, 0, 0, 1, 0, 0, 1})

	assert.Equal(t, expected, r)
}

func TestOutputDecode(t *testing.T) {
	vocab := map[string]int{"cat1": 0, "cat2": 1, "cat3": 2}
	s, p := OutputDecode(mat.NewDense(1, 3, []float64{0.5, 0.6, 0.2}), vocab)

	assert.Equal(t, "cat2", s)
	assert.Equal(t, 0.6, p)
}

func TestLevenshtein(t *testing.T) {
	w1 := "helo"
	w2 := "hello"
	d := Levenshtein(&w1, &w2)
	assert.Equal(t, 1, d)
}
