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

type Adam struct {
	learningRate float64
	beta         float64
	beta2        float64
	epsilon      float64

	v, m []float64
}

func NewAdam(lr, beta, beta2, epsilon float64) *Adam {
	return &Adam{
		learningRate: fparam(lr, 0.001),
		beta:         fparam(beta, 0.9),
		beta2:        fparam(beta2, 0.999),
		epsilon:      fparam(epsilon, 1e-8),
	}
}

func (o *Adam) UpdateParameters(value, gradient float64, t, idx int) {
	// lrt := o.lr * (math.Sqrt(1.0 - math.Pow(o.beta2, float64(t)))) /
	// 	(1.0 - math.Pow(o.beta, float64(t)))
	// o.m[idx] = o.beta*o.m[idx] + (1.0-o.beta)*gradient
	// o.v[idx] = o.beta2*o.v[idx] + (1.0-o.beta2)*math.Pow(gradient, 2.0)

	// return -lrt * (o.m[idx] / (math.Sqrt(o.v[idx]) + o.epsilon))
}

func fparam(val, fallback float64) float64 {
	if val == 0.0 {
		return fallback
	}
	return val
}
