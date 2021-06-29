package neunet

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestSoftmaxActivation(t *testing.T) {
	a := NewSoftmaxActivation()

	inputs := mat.NewDense(2, 3, []float64{-1.0, 0.0, 1.0, -0.5, 0.0, 0.5})
	a.Forward(inputs)

	assert.True(t, mat.Equal(a.inputs, inputs))

	res := mat.NewDense(2, 3, []float64{
		0.09003057317038046, 0.24472847105479764, 0.6652409557748218,
		0.1863237232258476, 0.3071958857184984, 0.506480391055654,
	})

	assert.True(t, mat.Equal(a.output, res))
}
