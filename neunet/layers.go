package neunet

import "gonum.org/v1/gonum/mat"

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
