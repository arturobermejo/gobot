package fn

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestDotTensors(t *testing.T) {
	x := NewTensor(mat.NewDense(2, 2, []float64{2., 4., 6., 8.}), true)
	y := NewTensor(mat.NewDense(2, 2, []float64{1., 0., -1., 0.}), true)
	z := Dot(x, y)

	assert.True(t, IsEqual(
		z, NewTensor(mat.NewDense(2, 2, []float64{-2., 0., -2., 0.}), false),
	))

	dinput := NewTensor(mat.NewDense(2, 2, []float64{1., 2., 0.1, 0.2}), false)

	z.Backward(dinput)

	assert.True(t, IsEqual(
		x.Grad, NewTensor(mat.NewDense(2, 2, []float64{1., -1., 0.1, -0.1}), false),
	))

	assert.True(t, IsEqual(
		y.Grad, NewTensor(mat.NewDense(2, 2, []float64{2.6, 5.2, 4.8, 9.6}), false),
	))
}
