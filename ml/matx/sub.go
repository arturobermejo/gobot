package matx

import (
	"log"

	"gonum.org/v1/gonum/mat"
)

func Sub(m, n mat.Matrix) *mat.Dense {
	var o *mat.Dense
	mr, mc := m.Dims()
	nr, nc := n.Dims()

	if mr == nr && mc == nc {
		o = mat.NewDense(mr, mc, nil)
		o.Sub(m, n)
	} else if mr == 1 && mc == nc {
		o = mat.NewDense(nr, nc, nil)
		o.Apply(func(i int, j int, v float64) float64 {
			return v - m.At(0, j)
		}, n)
	} else if nr == 1 && mc == nc {
		o = mat.NewDense(mr, mc, nil)
		o.Apply(func(i int, j int, v float64) float64 {
			return v - n.At(0, j)
		}, m)
	} else if mr == 1 && mc == 1 {
		o = mat.NewDense(nr, nc, nil)
		o.Apply(func(i int, j int, v float64) float64 {
			return v - m.At(0, 0)
		}, n)
	} else if nr == 1 && nc == 1 {
		o = mat.NewDense(mr, mc, nil)
		o.Apply(func(i int, j int, v float64) float64 {
			return v - n.At(0, 0)
		}, m)
	} else {
		log.Fatalf("Mismatch dimensions (%d, %d) (%d, %d).", mr, mc, nr, nc)
	}

	return o
}
