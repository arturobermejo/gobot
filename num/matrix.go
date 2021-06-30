package num

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

func Mul(m, n mat.Matrix) *mat.Dense {
	r, c := m.Dims()
	result := mat.NewDense(r, c, nil)
	result.MulElem(m, n)
	return result
}

func Div(m, n mat.Matrix) *mat.Dense {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.DivElem(m, n)
	return o
}

func Dot(m, n mat.Matrix) *mat.Dense {
	r, _ := m.Dims()
	_, c := n.Dims()
	o := mat.NewDense(r, c, nil)
	o.Product(m, n)
	return o
}

func Sqrt(m mat.Matrix) *mat.Dense {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)

	o.Apply(func(i int, j int, v float64) float64 {
		return math.Sqrt(v)
	}, m)

	return o
}

func MulScalar(s float64, m mat.Matrix) *mat.Dense {
	r, c := m.Dims()
	result := mat.NewDense(r, c, nil)
	result.Scale(s, m)
	return result
}

func Sum(m, n mat.Matrix) *mat.Dense {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)

	nr, _ := n.Dims()

	if nr == 1 {
		o.Apply(func(i int, j int, v float64) float64 {
			return v + n.At(0, j)
		}, m)
	} else {
		o.Add(m, n)
	}

	return o
}

func Sub(m, n mat.Matrix) *mat.Dense {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)

	nr, _ := n.Dims()

	if nr == 1 {
		o.Apply(func(i int, j int, v float64) float64 {
			return v - n.At(0, j)
		}, m)
	} else {
		o.Sub(m, n)
	}

	return o
}

func DivScalar(s float64, m mat.Matrix) *mat.Dense {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)

	o.Apply(func(i int, j int, v float64) float64 {
		return v / s
	}, m)

	return o
}

func SumScalar(s float64, m mat.Matrix) *mat.Dense {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)

	o.Apply(func(i int, j int, v float64) float64 {
		return v + s
	}, m)

	return o
}

func PowScalar(s float64, m mat.Matrix) *mat.Dense {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)

	o.Apply(func(i int, j int, v float64) float64 {
		return math.Pow(v, s)
	}, m)

	return o
}

func Print(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}

func Clip(m *mat.Dense, s, e float64) *mat.Dense {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)

	o.Apply(func(i int, j int, v float64) float64 {
		return math.Min(math.Max(s, v), e)
	}, m)

	return o
}
