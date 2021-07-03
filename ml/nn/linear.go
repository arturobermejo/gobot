package nn

import (
	"github.com/arturobermejo/gobot/ml/fn"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

func RandomSlice(size int) []float64 {
	dist := distuv.Normal{
		Mu:    0.0,
		Sigma: 0.05,
	}

	r := make([]float64, size)
	for i := 0; i < size; i++ {
		r[i] = dist.Rand()
	}
	return r
}

type Linear struct {
	inputs   *fn.Tensor
	weights  *fn.Tensor
	biases   *fn.Tensor
	Out      *fn.Tensor
	dinputs  *fn.Tensor
	dweights *fn.Tensor
	dbiases  *fn.Tensor

	// Parameters usded for Adam optimizer
	mweights *fn.Tensor
	mbiases  *fn.Tensor
	cweights *fn.Tensor
	cbiases  *fn.Tensor
}

func NewLinear(nInputs, nNeurons int) *Linear {
	return &Linear{
		weights: fn.NewTensor(mat.NewDense(nNeurons, nInputs, RandomSlice(nInputs*nNeurons)), true),
		biases:  fn.NewTensor(mat.NewDense(1, nNeurons, nil), true),

		mweights: fn.NewTensor(mat.NewDense(nInputs, nNeurons, nil), true),
		mbiases:  fn.NewTensor(mat.NewDense(1, nNeurons, nil), true),
		cweights: fn.NewTensor(mat.NewDense(nInputs, nNeurons, nil), true),
		cbiases:  fn.NewTensor(mat.NewDense(1, nNeurons, nil), true),
	}
}

func (l *Linear) Forward(inputs *fn.Tensor) {
	l.inputs = inputs
	l.Out = fn.Add(fn.Dot(inputs, fn.Trans(l.weights)), l.biases)
}

func (l *Linear) Backward(dvalues *fn.Tensor) {
	l.Out.Backward(dvalues)
	l.dbiases = l.biases.Grad
	l.dweights = l.weights.Grad
	l.dinputs = l.inputs.Grad
}
