package fn

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestTrans(t *testing.T) {
	x := NewTensor(mat.NewDense(3, 2, []float64{0.6, -0.1, -0.2, 0.2, 0.4, 0.6}), true)
	y := Trans(x)

	assert.True(t, IsEqual(
		y, NewTensor(mat.NewDense(2, 3, []float64{0.6, -0.2, 0.4, -0.1, 0.2, 0.6}), false),
	))

	grad := NewTensor(mat.NewDense(2, 3, []float64{1., 2., 3., 0.1, 0.2, 0.3}), false)
	y.Backward(grad)

	assert.True(t, IsEqual(
		x.Grad,
		NewTensor(mat.NewDense(3, 2, []float64{1., .1, 2., .2, 3., .3}), false),
	))
}
