package fn

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type reluOp struct {
	x   *Tensor
	out *Tensor
}

func ReLU(x *Tensor) *Tensor {
	op := reluOp{}
	op.out = op.Forward(x, nil)
	op.out.gradFn = &op
	return op.out
}

func (op *reluOp) Forward(x, y *Tensor) *Tensor {
	op.x = x
	reqGrad := x.ReqGrad
	return NewTensor(relu(x.Data), reqGrad)
}

func (op *reluOp) Backward(grad *Tensor) {
	if op.x.ReqGrad {
		// TODO: Refactor this part
		r, c := grad.Dims()
		o := mat.NewDense(r, c, nil)
		o.Apply(func(i, j int, v float64) float64 {
			if op.x.At(i, j) <= 0.0 {
				return 0.0
			}
			return v
		}, grad)

		op.x.addGrad(NewTensor(o, false))

		if op.x.gradFn != nil {
			op.x.Backward(nil)
		}
	}

}

func relu(m mat.Matrix) *mat.Dense {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)

	o.Apply(func(i int, j int, v float64) float64 {
		return math.Max(0, v)
	}, m)

	return o
}
