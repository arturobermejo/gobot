package neunet

import (
	"math"

	"gonum.org/v1/gonum/mat"
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

func softmax(matrix *mat.Dense) *mat.Dense {
	var sum float64
	// Calculate the sum
	for _, v := range matrix.RawMatrix().Data {
		sum += math.Exp(v)
	}

	resultMatrix := mat.NewDense(matrix.RawMatrix().Rows, matrix.RawMatrix().Cols, nil)
	// Calculate softmax value for each element
	resultMatrix.Apply(func(i int, j int, v float64) float64 {
		return math.Exp(v) / sum
	}, matrix)

	return resultMatrix
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
	o.Add(m, n)
	return o
}