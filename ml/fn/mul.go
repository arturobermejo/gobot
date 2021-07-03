package fn

import (
	"github.com/arturobermejo/gobot/ml/matx"
)

type mulOp struct {
	x   *Tensor
	y   *Tensor
	out *Tensor
}

func Mul(x, y *Tensor) *Tensor {
	op := mulOp{}
	op.out = op.Forward(x, y)
	op.out.gradFn = &op
	return op.out
}

func (op *mulOp) Forward(x, y *Tensor) *Tensor {
	op.x = x
	op.y = y
	reqGrad := x.ReqGrad || y.ReqGrad
	return NewTensor(matx.Mul(x.Data, y.Data), reqGrad)
}

func (op *mulOp) Backward(grad *Tensor) {
	if op.x.ReqGrad {
		op.x.addGrad(NewTensor(matx.Mul(grad.Data, op.y.Data), false))

		if op.x.gradFn != nil {
			op.x.Backward(nil)
		}
	}

	if op.y.ReqGrad {
		op.y.addGrad(NewTensor(matx.Mul(grad.Data, op.x.Data), false))

		if op.y.gradFn != nil {
			op.y.Backward(nil)
		}
	}
}
