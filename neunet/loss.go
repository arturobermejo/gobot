package neunet

import (
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

type CrossEntropy struct{}

func NewCrossEntropy() *CrossEntropy {
	return &CrossEntropy{}
}

func (c *CrossEntropy) Fordward(y, yPred *mat.Dense) float64 {
	return stat.CrossEntropy(y.RawMatrix().Data, yPred.RawMatrix().Data)
}
