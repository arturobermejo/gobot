package neunet

import (
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/gonum/stat/distuv"
)

func randomArray(size int) (data []float64) {
	dist := distuv.Normal{
		Mu:    0.0,
		Sigma: 0.05,
	}

	data = make([]float64, size)
	for i := 0; i < size; i++ {
		data[i] = dist.Rand()
	}
	return data
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
