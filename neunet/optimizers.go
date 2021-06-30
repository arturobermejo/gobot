package neunet

import (
	"math"

	"github.com/arturobermejo/gobot/num"
	"gonum.org/v1/gonum/mat"
)

type SGD struct {
	learningRate        float64
	currentLearningRate float64
	decay               float64
	iterations          int
	momentum            float64
}

func NewSGD(learningRate float64, decay float64, momentum float64) *SGD {
	return &SGD{
		learningRate:        learningRate,
		currentLearningRate: learningRate,
		decay:               decay,
		iterations:          0,
		momentum:            momentum,
	}
}

func (o *SGD) PreUpdateParams() {
	if o.decay != 0 {
		o.currentLearningRate = o.learningRate * (1.0 / (1 + o.decay*float64(o.iterations)))
	}
}

func (o *SGD) PostUpdateParams() {
	o.iterations += 1
}

func (o *SGD) UpdateParams(l *LinearLayer) {
	var dw, db *mat.Dense

	if o.momentum != 0.0 {
		dw = num.Sub(
			num.MulScalar(o.momentum, l.mweights),
			num.MulScalar(o.currentLearningRate, l.dweights),
		)
		l.mweights = dw

		db = num.Sub(
			num.MulScalar(o.momentum, l.mbiases),
			num.MulScalar(o.currentLearningRate, l.dbiases),
		)
		l.mbiases = db
	} else {
		dw = num.MulScalar(-1*o.currentLearningRate, l.dweights)
		db = num.MulScalar(-1*o.currentLearningRate, l.dbiases)
	}

	l.weights = num.Sum(l.weights, dw)
	l.biases = num.Sum(l.biases, db)
}

type Adam struct {
	learningRate        float64
	currentLearningRate float64
	decay               float64
	epsilon             float64
	beta1               float64
	beta2               float64
	iterations          int
	v, m                []float64
}

func NewAdam(lr float64) *Adam {
	return &Adam{
		learningRate:        lr,
		currentLearningRate: lr,
		decay:               0.0,
		epsilon:             1e-7,
		beta1:               0.9,
		beta2:               0.999,
	}
}

func (o *Adam) PreUpdateParams() {
	if o.decay != 0.0 {
		o.currentLearningRate = o.learningRate * (1.0 / (1 + o.decay*float64(o.iterations)))
	}
}

func (o *Adam) PostUpdateParams() {
	o.iterations += 1
}

func (o *Adam) UpdateParams(l *LinearLayer) {
	l.mweights = num.Sum(
		num.MulScalar(o.beta1, l.mweights), num.MulScalar((1-o.beta1), l.dweights),
	)
	l.mbiases = num.Sum(
		num.MulScalar(o.beta1, l.mbiases), num.MulScalar((1-o.beta1), l.dbiases),
	)

	mweightsCorr := num.DivScalar(1-math.Pow(o.beta1, float64(o.iterations+1)), l.mweights)
	mbiasesCorr := num.DivScalar(1-math.Pow(o.beta1, float64(o.iterations+1)), l.mbiases)

	l.cweights = num.Sum(
		num.MulScalar(o.beta2, l.cweights),
		num.MulScalar(1-o.beta2, num.PowScalar(2, l.dweights)),
	)
	l.cbiases = num.Sum(
		num.MulScalar(o.beta2, l.cbiases),
		num.MulScalar(1-o.beta2, num.PowScalar(2, l.dbiases)),
	)

	cweightsCorr := num.DivScalar(1-math.Pow(o.beta2, float64(o.iterations+1)), l.cweights)
	cbiasesCorr := num.DivScalar(1-math.Pow(o.beta2, float64(o.iterations+1)), l.cbiases)

	dw := num.Div(
		num.MulScalar(-1*o.currentLearningRate, mweightsCorr),
		num.SumScalar(o.epsilon, num.Sqrt(cweightsCorr)),
	)
	db := num.Div(
		num.MulScalar(-1*o.currentLearningRate, mbiasesCorr),
		num.SumScalar(o.epsilon, num.Sqrt(cbiasesCorr)),
	)

	l.weights = num.Sum(l.weights, dw)
	l.biases = num.Sum(l.biases, db)
}
