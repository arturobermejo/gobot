package nn

import (
	"github.com/arturobermejo/gobot/ml/fn"
)

type ReLU struct {
	inputs  *fn.Tensor
	Out     *fn.Tensor
	dinputs *fn.Tensor
}

func NewReLU() *ReLU {
	return &ReLU{}
}

func (a *ReLU) Forward(inputs *fn.Tensor) {
	a.inputs = inputs
	a.Out = fn.ReLU(inputs)
}

func (a *ReLU) Backward(dvalues *fn.Tensor) {
	a.inputs.Backward(dvalues)
	a.dinputs = a.inputs.Grad
}
