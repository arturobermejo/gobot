package neunet

import (
	"math"

	"github.com/arturobermejo/gobot/num"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

type CrossEntropy struct {
	inputs  *mat.Dense
	output  *mat.Dense
	dinputs *mat.Dense
}

func NewCrossEntropy() *CrossEntropy {
	return &CrossEntropy{}
}

func (l *CrossEntropy) Forward(inputs, yTrue *mat.Dense) float64 {
	l.inputs = inputs
	l.output = Softmax(l.inputs)

	o := nll(l.output, yTrue)
	return stat.Mean(o.RawMatrix().Data, nil)
}

func (l *CrossEntropy) Backward(dvalues *mat.Dense) {
	r, c := l.output.Dims()
	o := mat.NewDense(r, c, nil)
	o.CloneFrom(l.output)

	for i := 0; i < r; i++ {
		j := int(dvalues.At(0, i))
		v := l.output.At(i, j)
		o.Set(i, j, v-1)
	}

	l.dinputs = o
}

func nll(m, n mat.Matrix) *mat.Dense {
	mClipped := num.Clip(m.(*mat.Dense), 1e-7, 1-1e-7)
	r, c := n.Dims() // TODO: Validate r == 1, and mr == nc
	o := mat.NewDense(r, c, nil)

	for i := 0; i < c; i++ {
		o.Set(0, i, -1*math.Log(mClipped.At(i, int(n.At(0, i)))))
	}

	return o
}

func Softmax(m mat.Matrix) *mat.Dense {
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
