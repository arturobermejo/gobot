package num

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestMul(t *testing.T) {
	r := Mul(
		mat.NewDense(2, 3, []float64{-1, 2, 7, 0, -2, 9}),
		mat.NewDense(2, 3, []float64{2, -2, 4, 1, -4, 1}),
	)

	assert.True(t, mat.Equal(
		r, mat.NewDense(2, 3, []float64{-2, -4, 28, 0, 8, 9}),
	))
}

func TestDiv(t *testing.T) {
	r := Div(
		mat.NewDense(2, 3, []float64{4, 6, 8, 10, 2, 1}),
		mat.NewDense(2, 3, []float64{2, 3, 2, 2, 2, 2}),
	)

	assert.True(t, mat.Equal(
		r, mat.NewDense(2, 3, []float64{2, 2, 4, 5, 1, 0.5}),
	))
}

func TestDot(t *testing.T) {
	r := Dot(
		mat.NewDense(2, 3, []float64{-1, 2, 7, 0, -2, 9}),
		mat.NewDense(3, 2, []float64{2, -2, 4, 1, -4, 1}),
	)

	assert.True(t, mat.Equal(
		r, mat.NewDense(2, 2, []float64{-22, 11, -44, 7}),
	))
}

func TestSqrt(t *testing.T) {
	r := Sqrt(mat.NewDense(2, 3, []float64{4, 9, 16, 9, 16, 4}))

	assert.True(t, mat.Equal(
		r, mat.NewDense(2, 3, []float64{2, 3, 4, 3, 4, 2}),
	))
}

func TestMulScalar(t *testing.T) {
	r := MulScalar(2, mat.NewDense(2, 3, []float64{-1, 2, 7, 0, -2, 9}))

	assert.True(t, mat.Equal(
		r, mat.NewDense(2, 3, []float64{-2, 4, 14, 0, -4, 18}),
	))
}

func TestDivScalar(t *testing.T) {
	r := DivScalar(2, mat.NewDense(2, 3, []float64{-1, 2, 7, 0, -2, 9}))

	assert.True(t, mat.Equal(
		r, mat.NewDense(2, 3, []float64{-0.5, 1, 3.5, 0, -1, 4.5}),
	))
}

func TestSumScalar(t *testing.T) {
	r := SumScalar(2, mat.NewDense(2, 3, []float64{-1, 2, 7, 0, -2, 9}))

	assert.True(t, mat.Equal(
		r, mat.NewDense(2, 3, []float64{1, 4, 9, 2, 0, 11}),
	))
}

func TestPowScalar(t *testing.T) {
	r := PowScalar(2, mat.NewDense(2, 3, []float64{2, 2, 3, 3, 3, 2}))

	assert.True(t, mat.Equal(
		r, mat.NewDense(2, 3, []float64{4, 4, 9, 9, 9, 4}),
	))
}

func TestSum(t *testing.T) {
	r := Sum(
		mat.NewDense(2, 3, []float64{-1, 2, 7, 0, -2, 9}),
		mat.NewDense(1, 3, []float64{1, 2, 1}),
	)

	assert.True(t, mat.Equal(
		r, mat.NewDense(2, 3, []float64{0, 4, 8, 1, 0, 10}),
	))
}

func TestSub(t *testing.T) {
	r := Sub(
		mat.NewDense(2, 3, []float64{-1, 2, 7, 0, -2, 9}),
		mat.NewDense(1, 3, []float64{1, 2, 1}),
	)

	assert.True(t, mat.Equal(
		r, mat.NewDense(2, 3, []float64{-2, 0, 6, -1, -4, 8}),
	))
}
