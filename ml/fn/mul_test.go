package fn

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestMulTensors(t *testing.T) {
	x := NewTensor(mat.NewDense(1, 1, []float64{2.}), true)
	y := NewTensor(mat.NewDense(1, 1, []float64{3.}), true)
	z := Mul(x, y)

	assert.True(t, mat.Equal(
		z.Data, mat.NewDense(1, 1, []float64{6.}),
	))

	z.Backward(nil)

	assert.True(t, mat.Equal(
		x.Grad.Data, mat.NewDense(1, 1, []float64{3.}),
	))

	assert.True(t, mat.Equal(
		y.Grad.Data, mat.NewDense(1, 1, []float64{2.}),
	))
}
