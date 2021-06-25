package neunet

import (
	"math"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

type CrossEntropy struct{}

func NewCrossEntropy() *CrossEntropy {
	return &CrossEntropy{}
}

func (c *CrossEntropy) Fordward(yPred, y *mat.Dense) float64 {
	// TODO: clip to prevent division by 0

	correctConfidences := matrixMultiply(yPred, y)
	nr, _ := correctConfidences.Dims()

	o := mat.NewDense(nr, 1, nil)

	for i := 0; i < nr; i++ {
		o.Set(i, 0, -1*math.Log(mat.Sum(correctConfidences.RowView(i))))
	}

	mean := stat.Mean(o.RawMatrix().Data, nil)

	return mean
}
