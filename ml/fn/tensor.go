package fn

import (
	"log"

	"gonum.org/v1/gonum/mat"
)

type Op interface {
	Forward(x, y *Tensor) *Tensor
	Backward(grad *Tensor)
}

type Tensor struct {
	Data    mat.Matrix
	ReqGrad bool
	Grad    *Tensor
	gradFn  Op
}

func NewTensor(data mat.Matrix, reqGrad bool) *Tensor {
	return &Tensor{
		Data:    data,
		ReqGrad: reqGrad,
	}
}

func (ts *Tensor) Dims() (int, int) {
	return ts.Data.Dims()
}

func (ts *Tensor) At(i, j int) float64 {
	return ts.Data.At(i, j)
}

func (ts *Tensor) T() mat.Matrix {
	return trans(ts)
}

func (ts *Tensor) Backward(grad *Tensor) bool {
	if ts.gradFn == nil {
		return false
	}

	if grad == nil && ts.Grad == nil {
		grad = NewTensor(mat.NewDense(1, 1, []float64{1.}), false)
	} else if ts.Grad != nil {
		grad = ts.Grad
	}

	if !ts.ReqGrad {
		log.Fatal("This tensor is not backpropagated")
	}

	ts.gradFn.Backward(grad)

	return true
}

func (ts *Tensor) addGrad(grad *Tensor) {
	if ts.Grad == nil {
		ts.Grad = grad
	} else {
		// sum
	}
}
