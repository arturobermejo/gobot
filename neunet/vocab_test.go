package neunet

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestVocab(t *testing.T) {
	v := NewVocab()

	v.ProcessText("hello hello world ai")

	assert.Equal(t, v.Size(), 3)

	m := mat.NewDense(1, v.Size(), nil)
	v.OneHotEncode("bye bye world", m, 0, 0)

	assert.True(t, mat.Equal(
		m,
		mat.NewDense(1, v.Size(), []float64{0., 1., 0.}),
	))

	n := mat.NewDense(1, 1, nil)
	v.IndexEncode("world", n, 0)

	assert.True(t, mat.Equal(
		n,
		mat.NewDense(1, 1, []float64{1.}),
	))
}
