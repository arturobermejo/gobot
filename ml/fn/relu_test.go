package fn

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestReLUTensors(t *testing.T) {
	x := NewTensor(mat.NewDense(2, 2, []float64{-2., 2., -3., 3.}), true)
	y := ReLU(x)

	assert.True(t, IsEqual(
		y, NewTensor(mat.NewDense(2, 2, []float64{0., 2., 0., 3.}), false),
	))

	grad := NewTensor(mat.NewDense(2, 2, []float64{2., 2., 2., 2.}), false)
	y.Backward(grad)

	assert.True(t, IsEqual(
		x.Grad, NewTensor(mat.NewDense(2, 2, []float64{0., 2., 0., 2.}), false),
	))
}
