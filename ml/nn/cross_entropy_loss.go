package nn

import (
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"

	"github.com/arturobermejo/gobot/ml/fn"
)

type CrossEntropyLoss struct {
	inputs  *fn.Tensor
	output  *fn.Tensor
	dinputs *fn.Tensor
	meanRed bool
}

func NewCrossEntropyLoss(meanRed bool) *CrossEntropyLoss {
	return &CrossEntropyLoss{
		meanRed: meanRed,
	}
}

func (l *CrossEntropyLoss) Forward(inputs, yTrue *fn.Tensor) {
	l.inputs = inputs
	l.output = fn.CrossEntropy(l.inputs, yTrue)
}

func (l *CrossEntropyLoss) Backward(dvalues *fn.Tensor) {
	l.output.Backward(dvalues)
	l.dinputs = l.inputs.Grad
}

func (l *CrossEntropyLoss) Item() float64 {
	// TODO: Put this on tensors
	return stat.Mean(l.output.Data.(*mat.Dense).RawMatrix().Data, nil)
}
