package fn

import (
	"gonum.org/v1/gonum/mat"
)

type transOp struct {
	x   *Tensor
	out *Tensor
}

func Trans(x *Tensor) *Tensor {
	op := transOp{}
	op.out = op.Forward(x, nil)
	op.out.gradFn = &op
	return op.out
}

func (op *transOp) Forward(x, y *Tensor) *Tensor {
	op.x = x
	reqGrad := x.ReqGrad
	return NewTensor(trans(x.Data), reqGrad)
}

func (op *transOp) Backward(grad *Tensor) {
	if op.x.ReqGrad {
		op.x.addGrad(NewTensor(trans(grad.Data), false))

		if op.x.gradFn != nil {
			op.x.Backward(nil)
		}
	}
}

func trans(m mat.Matrix) mat.Matrix {
	return m.T()
}
