package neunet

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

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
