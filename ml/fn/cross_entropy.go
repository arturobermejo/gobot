package fn

import (
	"math"

	"github.com/arturobermejo/gobot/ml/matx"
	"gonum.org/v1/gonum/mat"
)

type crossEntropyOp struct {
	x   *Tensor
	y   *Tensor
	out *Tensor
}

func CrossEntropy(x, y *Tensor) *Tensor {
	op := crossEntropyOp{}
	op.out = op.Forward(x, y)
	op.out.gradFn = &op
	return op.out
}

func (op *crossEntropyOp) Forward(x, y *Tensor) *Tensor {
	op.x = x
	op.y = y
	reqGrad := x.ReqGrad || y.ReqGrad
	return NewTensor(crossEntropy(x.Data, y.Data), reqGrad)
}

func (op *crossEntropyOp) Backward(grad *Tensor) {
	// TODO: Determine how to use the grad argument
	s := softmax(op.x.Data)
	r, c := s.Dims()
	o := mat.NewDense(r, c, nil)
	o.CloneFrom(s)

	for i := 0; i < r; i++ {
		j := int(op.y.Data.At(0, i))
		v := s.At(i, j)
		o.Set(i, j, v-1)
	}

	if op.x.ReqGrad {
		op.x.addGrad(NewTensor(o, false))

		if op.x.gradFn != nil {
			op.x.Backward(nil)
		}
	}
}

func crossEntropy(m, n mat.Matrix) *mat.Dense {
	o := softmax(m)
	return nll(o, n)
}

func softmax(m mat.Matrix) *mat.Dense {
	r, c := m.Dims()

	expValues := mat.NewDense(r, c, nil)

	for i := 0; i < r; i++ {
		max := mat.Max(m.(*mat.Dense).RowView(i)) // TODO: Refact this casting

		for j := 0; j < c; j++ {
			v := math.Exp(m.At(i, j) - max)
			expValues.Set(i, j, v)
		}
	}

	o := mat.NewDense(r, c, nil)

	for i := 0; i < r; i++ {
		s := mat.Sum(expValues.RowView(i))

		for j := 0; j < c; j++ {
			v := expValues.At(i, j) / s
			o.Set(i, j, v)
		}
	}

	return o
}

func nll(m, n mat.Matrix) *mat.Dense {
	mClipped := matx.Clip(m.(*mat.Dense), 1e-7, 1-1e-7)
	r, c := n.Dims() // TODO: Validate r == 1, and mr == nc
	o := mat.NewDense(r, c, nil)

	for i := 0; i < c; i++ {
		o.Set(0, i, -1*math.Log(mClipped.At(i, int(n.At(0, i)))))
	}

	return o
}
