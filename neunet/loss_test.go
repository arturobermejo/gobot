package neunet

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestCrossEntropy(t *testing.T) {
	c := NewCrossEntropy()
	v := c.Forward(
		mat.NewDense(3, 2, []float64{.3, .4, .5, .6, .8, .3}),
		mat.NewDense(1, 3, []float64{0., 1., 1.}),
	)

	assert.Equal(t, 0.7876234347757496, v)
	assert.True(t, mat.Equal(
		c.output, mat.NewDense(3, 2, []float64{
			0.47502081252106, 0.52497918747894, 0.47502081252106,
			0.5249791874789399, 0.6224593312018546, 0.37754066879814546,
		}),
	))
}
