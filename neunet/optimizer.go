package neunet

import "gonum.org/v1/gonum/mat"

type Layer interface {
	Fordward(inputs *mat.Dense)
	Backward(dvalues *mat.Dense)
	GetWeights() *mat.Dense
	SetWeights(weights *mat.Dense)
	GetdWeights() *mat.Dense
}

type SGD struct {
	learningRate float64
}

func NewSGD(learningRate float64) *SGD {
	return &SGD{
		learningRate: learningRate,
	}
}

func (o *SGD) UpdateParameters(l Layer) {
	l.SetWeights(
		matrixSubtract(l.GetWeights(), matrixScale(o.learningRate, l.GetdWeights())),
	)
}
