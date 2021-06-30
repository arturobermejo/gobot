package neunet

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestAccuracy(t *testing.T) {
	acc := Accuracy(
		mat.NewDense(2, 3, []float64{0.9, 0.1, 0.5, 0.2, 0.4, 0.7}),
		mat.NewDense(2, 3, []float64{1, 0, 0, 0, 1, 0}),
	)

	assert.Equal(t, 0.5, acc)
}

func TestArgmax(t *testing.T) {
	idx, v := Argmax([]float64{-1.0, 9.0, 5.0, 2.0, 0.5, -10.0})

	assert.Equal(t, 1, idx)
	assert.Equal(t, 9.0, v)
}

func TestCleanText(t *testing.T) {
	s := CleanText("12?Hel,Lo ?45wórld.")

	assert.Equal(t, "hello world", s)
}
