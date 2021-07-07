package neunet

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestDataLoader(t *testing.T) {
	x := mat.NewDense(5, 1, []float64{1., 2., 3., 4., 5.})
	y := mat.NewDense(1, 5, []float64{1., 2., 3., 4., 5})

	dl := NewDataLoader(x, y, 2)

	assert.Equal(t, 3, dl.BatchCount())
	assert.True(t, dl.Batch())

	input, output := dl.Data()

	assert.True(t, mat.Equal(
		input,
		mat.NewDense(2, 1, []float64{1., 2.}),
	))

	assert.True(t, mat.Equal(
		output,
		mat.NewDense(1, 2, []float64{1., 2.}),
	))

	input, output = dl.Data()

	assert.True(t, mat.Equal(
		input,
		mat.NewDense(2, 1, []float64{3., 4.}),
	))

	assert.True(t, mat.Equal(
		output,
		mat.NewDense(1, 2, []float64{3., 4.}),
	))

	input, output = dl.Data()

	assert.True(t, mat.Equal(
		input,
		mat.NewDense(1, 1, []float64{5.}),
	))

	assert.True(t, mat.Equal(
		output,
		mat.NewDense(1, 1, []float64{5.}),
	))

	assert.False(t, dl.Batch())
}
