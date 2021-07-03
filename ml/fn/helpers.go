package fn

import "gonum.org/v1/gonum/mat"

func MaxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func IsEqual(m, n *Tensor) bool {
	// TODO: Verify reqGrad
	return mat.Equal(m.Data, n.Data)
}

func SumCol(m *Tensor) *Tensor {
	_, c := m.Data.Dims()
	o := mat.NewDense(1, c, nil)

	for i := 0; i < c; i++ {
		s := mat.Sum(m.Data.(*mat.Dense).ColView(i))
		o.Set(0, i, s)
	}

	return NewTensor(o, m.ReqGrad)
}
