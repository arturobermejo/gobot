package fn

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestCrossEntropyTensors(t *testing.T) {
	yPred := NewTensor(mat.NewDense(3, 2, []float64{.3, .4, .5, .6, .8, .3}), true)
	yTrue := NewTensor(mat.NewDense(1, 3, []float64{0., 1., 1.}), false)
	z := CrossEntropy(yPred, yTrue)

	assert.True(t, IsEqual(
		z, NewTensor(mat.NewDense(1, 3, []float64{
			0.744396660073571, 0.6443966600735711, 0.9740769841801067,
		}), false),
	))

	grad := NewTensor(mat.NewDense(1, 3, []float64{0., 1., 1.}), false)
	z.Backward(grad)

	assert.True(t, IsEqual(
		yPred.Grad, NewTensor(mat.NewDense(3, 2, []float64{
			-0.52497918747894, 0.52497918747894,
			0.47502081252106, -0.4750208125210601,
			0.6224593312018546, -0.6224593312018545,
		}), false),
	))
}
