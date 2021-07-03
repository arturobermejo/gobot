package fn

import (
	"github.com/arturobermejo/gobot/ml/matx"
)

type addOp struct {
	x   *Tensor
	y   *Tensor
	out *Tensor
}

func Add(x, y *Tensor) *Tensor {
	op := addOp{}
	op.out = op.Forward(x, y)
	op.out.gradFn = &op
	return op.out
}

func (op *addOp) Forward(x, y *Tensor) *Tensor {
	op.x = x
	op.y = y
	reqGrad := x.ReqGrad || y.ReqGrad
	return NewTensor(matx.Add(x.Data, y.Data), reqGrad)
}

func (op *addOp) Backward(grad *Tensor) {
	gr, gc := grad.Data.Dims()

	if op.x.ReqGrad {
		xr, xc := op.x.Data.Dims()

		if gr != xr || gc != xc {
			op.x.addGrad(SumCol(grad))
		} else {
			op.x.addGrad(grad)
		}

		if op.x.gradFn != nil {
			op.x.Backward(nil)
		}
	}

	if op.y.ReqGrad {
		yr, yc := op.y.Data.Dims()

		if gr != yr || gc != yc {
			op.y.addGrad(SumCol(grad))
		} else {
			op.y.addGrad(grad) // grad * 1
		}

		if op.y.gradFn != nil {
			op.y.Backward(nil)
		}
	}
}
