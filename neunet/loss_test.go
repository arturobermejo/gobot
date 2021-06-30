package neunet

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestCrossEntropy(t *testing.T) {
	c := NewCrossEntropy()
	v := c.Forward(
		mat.NewDense(3, 2, []float64{0.3, 0.7, 0.1, 0.9, 0.2, 0.8}),
		mat.NewDense(3, 2, []float64{1, 0, 0, 1, 0, 1}),
	)

	assert.Equal(t, 0.5108256237659907, v)
}
