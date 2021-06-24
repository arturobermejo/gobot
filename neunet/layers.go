package neunet

import "gonum.org/v1/gonum/mat"

type DenseLayer struct {
	weights *mat.Dense
	biases  *mat.Dense
	output  *mat.Dense
}

func NewDenseLayer(nInputs, nNeurons int) *DenseLayer {
	return &DenseLayer{
		weights: mat.NewDense(nInputs, nNeurons, randomArray(nInputs*nNeurons, float64(nInputs))),
		biases:  mat.NewDense(1, nNeurons, nil),
	}
}

func (l *DenseLayer) Fordward(inputs *mat.Dense) {
	l.output = matrixAdd(matrixDot(inputs, l.weights), l.biases)
}
