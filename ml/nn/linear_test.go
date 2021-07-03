package nn

import (
	"testing"

	"github.com/arturobermejo/gobot/ml/fn"
	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestLinear(t *testing.T) {
	l := NewLinear(2, 3)

	l.weights = fn.NewTensor(mat.NewDense(3, 2, []float64{0.6, -0.1, -0.2, 0.2, 0.4, 0.6}), true)
	l.biases = fn.NewTensor(mat.NewDense(1, 3, []float64{-0.1, 0.4, -0.5}), true)

	input := fn.NewTensor(mat.NewDense(2, 2, []float64{2.0, 2.0, 4.0, 4.0}), true)
	l.Forward(input)

	assert.True(t, fn.IsEqual(
		l.Out,
		fn.NewTensor(mat.NewDense(2, 3, []float64{0.9, 0.4, 1.5, 1.9, 0.4, 3.5}), false),
	))

	dinput := fn.NewTensor(mat.NewDense(2, 3, []float64{1., 2., 3., 0.1, 0.2, 0.3}), false)

	l.Backward(dinput)

	assert.True(t, fn.IsEqual(
		l.dinputs,
		fn.NewTensor(mat.NewDense(2, 2, []float64{1.4000000000000001, 2.0999999999999996, 0.13999999999999999, 0.21}), false),
	))

	assert.True(t, fn.IsEqual(
		l.dweights,
		fn.NewTensor(mat.NewDense(3, 2, []float64{2.4, 2.4, 4.8, 4.8, 7.2, 7.2}), false),
	))

	assert.True(t, fn.IsEqual(
		l.dbiases,
		fn.NewTensor(mat.NewDense(1, 3, []float64{1.1, 2.2, 3.3}), false),
	))
}
