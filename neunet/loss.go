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

func (c *CrossEntropy) Forward(yPred, yTrue *mat.Dense) float64 {
	yPredClipped := matrixClip(yPred, 1e-7, 1-1e-7)

	e := matrixMultiply(yPredClipped, yTrue)
	r, _ := e.Dims()
	o := mat.NewDense(1, r, nil)

	for i := 0; i < r; i++ {
		o.Set(0, i, -1*math.Log(mat.Sum(e.RowView(i))))
	}

	mean := stat.Mean(o.RawMatrix().Data, nil)

	return mean
}
