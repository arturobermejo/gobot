package neunet

import (
	"math"

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
		dw = matrixSubtract(
			matrixScale(o.momentum, l.mweights),
			matrixScale(o.currentLearningRate, l.dweights),
		)
		l.mweights = dw

		db = matrixSubtract(
			matrixScale(o.momentum, l.mbiases),
			matrixScale(o.currentLearningRate, l.dbiases),
		)
		l.mbiases = db
	} else {
		dw = matrixScale(-1*o.currentLearningRate, l.dweights)
		db = matrixScale(-1*o.currentLearningRate, l.dbiases)
	}

	l.weights = matrixAdd(l.weights, dw)
	l.biases = matrixAdd(l.biases, db)
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
	// layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights

	l.mweights = matrixAdd(
		matrixScale(o.beta1, l.mweights), matrixScale((1-o.beta1), l.dweights),
	)
	// layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
	l.mbiases = matrixAdd(
		matrixScale(o.beta1, l.mbiases), matrixScale((1-o.beta1), l.dbiases),
	)

	// weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
	mweightsCorr := matrixDivScale(1-math.Pow(o.beta1, float64(o.iterations+1)), l.mweights)
	// bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))
	mbiasesCorr := matrixDivScale(1-math.Pow(o.beta1, float64(o.iterations+1)), l.mbiases)

	// layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
	l.cweights = matrixAdd(
		matrixScale(o.beta2, l.cweights),
		matrixScale(1-o.beta2, matrixPowScale(2, l.dweights)),
	)
	// layer.bias_cache = self.beta_2 * layer.bias_cache + \ (1 - self.beta_2) * layer.dbiases**2
	l.cbiases = matrixAdd(
		matrixScale(o.beta2, l.cbiases),
		matrixScale(1-o.beta2, matrixPowScale(2, l.dbiases)),
	)

	// weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
	cweightsCorr := matrixDivScale(1-math.Pow(o.beta2, float64(o.iterations+1)), l.cweights)
	// bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))
	cbiasesCorr := matrixDivScale(1-math.Pow(o.beta2, float64(o.iterations+1)), l.cbiases)

	// -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
	dw := matrixDiv(
		matrixScale(-1*o.currentLearningRate, mweightsCorr),
		matrixScale(o.epsilon, matrixSqrt(cweightsCorr)),
	)
	// -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)
	db := matrixDiv(
		matrixScale(-1*o.currentLearningRate, mbiasesCorr),
		matrixScale(o.epsilon, matrixSqrt(cbiasesCorr)),
	)

	l.weights = matrixAdd(l.weights, dw)
	l.biases = matrixAdd(l.biases, db)
}
