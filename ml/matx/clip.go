package matx

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

func Clip(m mat.Matrix, s, e float64) *mat.Dense {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)

	o.Apply(func(i int, j int, v float64) float64 {
		return math.Min(math.Max(s, v), e)
	}, m)

	return o
}
