package neunet

import (
	"gonum.org/v1/gonum/mat"
)

type Layer interface {
	Forward(inputs *mat.Dense)
	Backward(dvalues *mat.Dense)
	GetWeights() *mat.Dense
	SetWeights(weights *mat.Dense)
	GetdWeights() *mat.Dense
	GetBiases() *mat.Dense
	SetBiases(biases *mat.Dense)
	GetdBiases() *mat.Dense
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
	dw := matrixScale(o.learningRate, l.GetdWeights())
	db := matrixScale(o.learningRate, l.GetdBiases())
	l.SetWeights(matrixSubtract(l.GetWeights(), dw))
	l.SetBiases(matrixSubtract(l.GetBiases(), db))
}
