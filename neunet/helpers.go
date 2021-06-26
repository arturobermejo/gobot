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

func matrixSubtract(m, n mat.Matrix) *mat.Dense {
	r, c := m.Dims()
	result := mat.NewDense(r, c, nil)
	result.Sub(m, n)
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

func matrixScale(s float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	result := mat.NewDense(r, c, nil)
	result.Scale(s, m)
	return result
}

func randomArray(size int, v float64) (data []float64) {
	dist := distuv.Uniform{
		Min: -1 / math.Sqrt(v),
		Max: 1 / math.Sqrt(v),
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

func matrixPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}

func accuracy(m, n *mat.Dense) float64 {
	r, _ := m.Dims()
	result := mat.NewDense(1, r, nil)

	for i := 0; i < r; i++ {
		mi := argmax(m.RawRowView(i))
		ni := argmax(n.RawRowView(i))

		if mi == ni {
			result.Set(0, mi, 1)
		}
	}

	return stat.Mean(result.RawMatrix().Data, nil)
}

func argmax(s []float64) int {
	var maxIdx int

	n := len(s)
	max := s[0]
	maxIdx = 0

	for i := 1; i < n; i++ {
		v := s[i]
		if v > max {
			max = v
			maxIdx = i
			break
		}
	}

	return maxIdx
}
