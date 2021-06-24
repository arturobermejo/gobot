package neunet

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type LinearLayer struct {
	weights *mat.Dense
	biases  *mat.Dense
	output  *mat.Dense
}

func NewLinearLayer(nInputs, nNeurons int) *LinearLayer {
	return &LinearLayer{
		weights: mat.NewDense(nInputs, nNeurons, randomArray(nInputs*nNeurons, float64(nInputs))),
		biases:  mat.NewDense(1, nNeurons, nil),
	}
}

func (l *LinearLayer) Fordward(inputs *mat.Dense) {
	l.output = matrixAdd(matrixDot(inputs, l.weights), l.biases)
}

type SoftmaxActivation struct {
	output *mat.Dense
}

func NewSoftmaxActivation() *SoftmaxActivation {
	return &SoftmaxActivation{}
}

func (a *SoftmaxActivation) Fordward(inputs *mat.Dense) {
	var sum float64
	for _, v := range inputs.RawMatrix().Data {
		sum += math.Exp(v)
	}

	m := mat.NewDense(inputs.RawMatrix().Rows, inputs.RawMatrix().Cols, nil)

	m.Apply(func(i int, j int, v float64) float64 {
		return math.Exp(v) / sum
	}, inputs)

	a.output = m
}

type ReLUActivation struct {
	output *mat.Dense
}

func NewReLUActivation() *ReLUActivation {
	return &ReLUActivation{}
}

func (a *ReLUActivation) Fordward(inputs *mat.Dense) {
	r, c := inputs.Dims()
	m := mat.NewDense(r, c, nil)

	m.Apply(func(i int, j int, v float64) float64 {
		return math.Max(0, v)
	}, inputs)

	a.output = m
}
