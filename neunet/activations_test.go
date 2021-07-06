package neunet

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestReluActivation(t *testing.T) {
	a := NewReLUActivation()

	// Forward
	inputs := mat.NewDense(2, 3, []float64{-1.0, 0.0, 1.0, -0.5, 0.0, 0.5})
	a.Forward(inputs)

	assert.True(t, mat.Equal(a.inputs, inputs))

	res := mat.NewDense(2, 3, []float64{0.0, 0.0, 1.0, 0.0, 0.0, 0.5})

	assert.True(t, mat.Equal(a.output, res))

	// Backward
	inputs = mat.NewDense(2, 3, []float64{1.0, 2.0, 3.0, -0.1, -0.2, -0.3})
	a.Backward(inputs)

	res = mat.NewDense(2, 3, []float64{0.0, 0.0, 3.0, 0.0, 0.0, -0.3})

	assert.True(t, mat.Equal(a.dinputs, res))
}
