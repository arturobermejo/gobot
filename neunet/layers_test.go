package neunet

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestLinearLayer(t *testing.T) {
	l := NewLinearLayer(2, 3)

	r, c := l.weights.Dims()
	assert.Equal(t, 2, r, nil)
	assert.Equal(t, 3, c, nil)

	r, c = l.biases.Dims()
	assert.Equal(t, 1, r, nil)
	assert.Equal(t, 3, c, nil)

	// Test Forward
	l.weights = mat.NewDense(2, 3, []float64{1, 2, 3, -1, -2, -3})
	l.biases = mat.NewDense(1, 3, []float64{-1, 1, -1})

	input := mat.NewDense(4, 2, []float64{2, 2, 3, 3, 4, 4, 5, 5})
	l.Forward(input)

	r, c = l.output.Dims()
	assert.Equal(t, 4, r, nil)
	assert.Equal(t, 3, c, nil)

	isEqual := mat.Equal(l.output, mat.NewDense(
		4, 3, []float64{-1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1}))
	assert.True(t, isEqual)

	// Test Backward
	l.weights = mat.NewDense(2, 2, []float64{1, 2, -1, -2})
	l.inputs = mat.NewDense(2, 2, []float64{0.5, 0.5, -0.5, -0.5})

	l.Backward(mat.NewDense(2, 2, []float64{2, 2, 2, 1}))

	assert.True(t, mat.Equal(
		l.dbiases, mat.NewDense(1, 2, []float64{4, 3}),
	))
	assert.True(t, mat.Equal(
		l.dweights, mat.NewDense(2, 2, []float64{0, 0.5, 0, 0.5}),
	))
	assert.True(t, mat.Equal(
		l.dinputs, mat.NewDense(2, 2, []float64{6.0, -6.0, 4.0, -4.0}),
	))
}
