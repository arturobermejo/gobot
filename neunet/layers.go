package neunet

import (
	"github.com/arturobermejo/gobot/num"
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

	// Parameters usded for Adam optimizer
	mweights *mat.Dense
	mbiases  *mat.Dense
	cweights *mat.Dense
	cbiases  *mat.Dense
}

func NewLinearLayer(nInputs, nNeurons int) *LinearLayer {
	return &LinearLayer{
		weights: mat.NewDense(nInputs, nNeurons, RandomSlice(nInputs*nNeurons)),
		biases:  mat.NewDense(1, nNeurons, nil),

		mweights: mat.NewDense(nInputs, nNeurons, nil),
		mbiases:  mat.NewDense(1, nNeurons, nil),
		cweights: mat.NewDense(nInputs, nNeurons, nil),
		cbiases:  mat.NewDense(1, nNeurons, nil),
	}
}

func (l *LinearLayer) Forward(inputs *mat.Dense) {
	l.inputs = inputs
	l.output = num.Sum(num.Dot(inputs, l.weights), l.biases)
}

func (l *LinearLayer) Backward(dvalues *mat.Dense) {
	_, c := dvalues.Dims()
	l.dbiases = mat.NewDense(1, c, nil)

	for i := 0; i < c; i++ {
		s := mat.Sum(dvalues.ColView(i))
		l.dbiases.Set(0, i, s)
	}

	l.dweights = num.Dot(l.inputs.T(), dvalues)
	l.dinputs = num.Dot(dvalues, l.weights.T())
}
