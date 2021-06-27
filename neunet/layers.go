package neunet

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type LinearLayer struct {
	inputs   *mat.Dense
	weights  *mat.Dense
	biases   *mat.Dense
	output   *mat.Dense
	dinputs  *mat.Dense
	dweights *mat.Dense
	dbiases  *mat.Dense
}

func NewLinearLayer(nInputs, nNeurons int) *LinearLayer {
	return &LinearLayer{
		weights: mat.NewDense(nInputs, nNeurons, randomArray(nInputs*nNeurons, float64(nInputs))),
		biases:  mat.NewDense(1, nNeurons, nil),
	}
}

func (l *LinearLayer) Forward(inputs *mat.Dense) {
	l.inputs = inputs
	e := matrixDot(inputs, l.weights)
	l.output = matrixAdd(e, l.biases)
}

func (l *LinearLayer) Backward(dvalues *mat.Dense) {
	// TODO: Put this in helpers
	_, c := dvalues.Dims()
	l.dbiases = mat.NewDense(1, c, nil)

	for i := 0; i < c; i++ {
		s := mat.Sum(dvalues.ColView(i))
		l.dbiases.Set(0, i, s)
	}

	l.dweights = matrixDot(l.inputs.T(), dvalues)
	l.dinputs = matrixDot(dvalues, l.weights.T())
}

func (l *LinearLayer) GetWeights() *mat.Dense {
	return l.weights
}

func (l *LinearLayer) SetWeights(weights *mat.Dense) {
	l.weights = weights
}

func (l *LinearLayer) GetdWeights() *mat.Dense {
	return l.dweights
}

func (l *LinearLayer) GetBiases() *mat.Dense {
	return l.biases
}

func (l *LinearLayer) SetBiases(biases *mat.Dense) {
	l.biases = biases
}

func (l *LinearLayer) GetdBiases() *mat.Dense {
	return l.dbiases
}

type SoftmaxActivation struct {
	output *mat.Dense
}

func NewSoftmaxActivation() *SoftmaxActivation {
	return &SoftmaxActivation{}
}

func (a *SoftmaxActivation) Forward(inputs *mat.Dense) {
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
