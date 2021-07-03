package fn

import (
	"github.com/arturobermejo/gobot/ml/matx"
)

type dotOp struct {
	x   *Tensor
	y   *Tensor
	out *Tensor
}

func Dot(x, y *Tensor) *Tensor {
	op := dotOp{}
	op.out = op.Forward(x, y)
	op.out.gradFn = &op
	return op.out
}

func (op *dotOp) Forward(x, y *Tensor) *Tensor {
	op.x = x
	op.y = y
	reqGrad := x.ReqGrad || y.ReqGrad
	return NewTensor(matx.Dot(x.Data, y.Data), reqGrad)
}

func (op *dotOp) Backward(grad *Tensor) {
	if op.x.ReqGrad {
		op.x.addGrad(NewTensor(matx.Dot(grad.Data, op.y.Data.T()), false))

		if op.x.gradFn != nil {
			op.x.Backward(nil)
		}
	}

	if op.y.ReqGrad {
		op.y.addGrad(NewTensor(matx.Dot(op.x.Data.T(), grad.Data), false))

		if op.y.gradFn != nil {
			op.y.Backward(nil)
		}
	}
}
