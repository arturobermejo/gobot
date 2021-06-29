package neunet

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type SoftmaxActivation struct {
	output *mat.Dense
}

func NewSoftmaxActivation() *SoftmaxActivation {
	return &SoftmaxActivation{}
}

func (a *SoftmaxActivation) Forward(inputs *mat.Dense) {
	r, c := inputs.Dims()

	expValues := mat.NewDense(r, c, nil)

	for i := 0; i < r; i++ {
		max := mat.Max(inputs.RowView(i))

		for j := 0; j < c; j++ {
			v := math.Exp(inputs.At(i, j) - max)
			expValues.Set(i, j, v)
		}
	}

	m := mat.NewDense(r, c, nil)

	for i := 0; i < r; i++ {
		s := mat.Sum(expValues.RowView(i))

		for j := 0; j < c; j++ {
			v := expValues.At(i, j) / s
			m.Set(i, j, v)
		}
	}

	a.output = m
}

type ReLUActivation struct {
	inputs  *mat.Dense
	output  *mat.Dense
	dinputs *mat.Dense
}

func NewReLUActivation() *ReLUActivation {
	return &ReLUActivation{}
}

func (a *ReLUActivation) Forward(inputs *mat.Dense) {
	a.inputs = inputs

	r, c := inputs.Dims()
	m := mat.NewDense(r, c, nil)

	m.Apply(func(i int, j int, v float64) float64 {
		return math.Max(0, v)
	}, inputs)

	a.output = m
}

func (a *ReLUActivation) Backward(dvalues *mat.Dense) {
	r, c := dvalues.Dims()
	a.dinputs = mat.NewDense(r, c, nil)
	a.dinputs.Apply(func(i, j int, v float64) float64 {
		if a.inputs.At(i, j) <= 0.0 {
			return 0.0
		}
		return v
	}, dvalues)
}