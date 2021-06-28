package neunet

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/gonum/stat/distuv"
)

func sigmoid(m *mat.Dense) *mat.Dense {
	f := func(i, j int, v float64) float64 {
		return 1 / (1 + math.Exp(-1*v))
	}

	r, c := m.Dims()
	result := mat.NewDense(r, c, nil)
	result.Apply(f, m)
	return result
}

func SigmoidOutputDerivative(m *mat.Dense) *mat.Dense {
	sigmoidDerivative := func(i, j int, v float64) float64 {
		return v * (1 - v)
	}
	r, c := m.Dims()
	result := mat.NewDense(r, c, nil)
	result.Apply(sigmoidDerivative, m)
	return result
}

func matrixMultiply(m, n mat.Matrix) *mat.Dense {
	r, c := m.Dims()
	result := mat.NewDense(r, c, nil)
	result.MulElem(m, n)
	return result
}

func matrixDot(m, n mat.Matrix) *mat.Dense {
	r, _ := m.Dims()
	_, c := n.Dims()
	o := mat.NewDense(r, c, nil)
	o.Product(m, n)
	return o
}

func matrixScale(s float64, m mat.Matrix) *mat.Dense {
	r, c := m.Dims()
	result := mat.NewDense(r, c, nil)
	result.Scale(s, m)
	return result
}

func matrixDiv(m, n mat.Matrix) *mat.Dense {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.DivElem(m, n)
	return o
}

func matrixSqrt(m mat.Matrix) *mat.Dense {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)

	o.Apply(func(i int, j int, v float64) float64 {
		return math.Sqrt(v)
	}, m)

	return o
}

func randomArray(size int) (data []float64) {
	dist := distuv.Normal{
		Mu:    0.0,
		Sigma: 0.05,
	}

	data = make([]float64, size)
	for i := 0; i < size; i++ {
		data[i] = dist.Rand()
	}
	return
}

func matrixAdd(m, n mat.Matrix) *mat.Dense {
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

func matrixSubtract(m, n mat.Matrix) *mat.Dense {
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

func matrixDivScale(s float64, m mat.Matrix) *mat.Dense {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)

	o.Apply(func(i int, j int, v float64) float64 {
		return v / s
	}, m)

	return o
}

func matrixAddScale(s float64, m mat.Matrix) *mat.Dense {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)

	o.Apply(func(i int, j int, v float64) float64 {
		return v + s
	}, m)

	return o
}

func matrixPowScale(s float64, m mat.Matrix) *mat.Dense {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)

	o.Apply(func(i int, j int, v float64) float64 {
		return math.Pow(v, s)
	}, m)

	return o
}

func matrixPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}

func accuracy(m, n *mat.Dense) float64 {
	r, _ := m.Dims()
	result := mat.NewDense(1, r, nil)

	for i := 0; i < r; i++ {
		mi := Argmax(m.RawRowView(i))
		ni := Argmax(n.RawRowView(i))

		if mi == ni {
			result.Set(0, i, 1)
		}
	}

	return stat.Mean(result.RawMatrix().Data, nil)
}

func Argmax(s []float64) int {
	var maxIdx int

	n := len(s)
	max := s[0]
	maxIdx = 0

	for i := 1; i < n; i++ {
		v := s[i]
		if v > max {
			max = v
			maxIdx = i
		}
	}

	return maxIdx
}

func matrixClip(m *mat.Dense, s, e float64) *mat.Dense {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)

	o.Apply(func(i int, j int, v float64) float64 {
		return math.Min(math.Max(s, v), e)
	}, m)

	return o
}
