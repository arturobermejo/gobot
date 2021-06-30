package neunet

import (
	"fmt"
	"os"

	"github.com/arturobermejo/gobot/num"
	"gonum.org/v1/gonum/mat"
)

type DataLoader interface {
	Sample(i, offset int) (*mat.Dense, *mat.Dense)
	Size() int
}

type Model struct {
	inputSize  int
	hiddenSize int
	outputSize int

	hiddenLayer  *LinearLayer
	reluLayer    *ReLUActivation
	outputLayer  *LinearLayer
	softmaxLayer *SoftmaxActivation
}

func NewModel(inputSize, outputSize int) *Model {
	hiddenSize := 32

	m := &Model{
		inputSize:    inputSize,
		hiddenSize:   hiddenSize,
		outputSize:   outputSize,
		hiddenLayer:  NewLinearLayer(inputSize, hiddenSize),
		reluLayer:    NewReLUActivation(),
		outputLayer:  NewLinearLayer(hiddenSize, outputSize),
		softmaxLayer: NewSoftmaxActivation(),
	}

	return m
}

func LoadModel(dir string) *Model {
	var hWeights, oWeights, hBiases, oBiases mat.Dense

	hw, err := os.Open(fmt.Sprintf("%s/hw.model", dir))
	defer hw.Close()
	if err == nil {
		hWeights.UnmarshalBinaryFrom(hw)
	}

	ow, err := os.Open(fmt.Sprintf("%s/ow.model", dir))
	defer ow.Close()
	if err == nil {
		oWeights.UnmarshalBinaryFrom(ow)
	}

	hb, err := os.Open(fmt.Sprintf("%s/hb.model", dir))
	defer hb.Close()
	if err == nil {
		hBiases.UnmarshalBinaryFrom(hb)
	}

	ob, err := os.Open(fmt.Sprintf("%s/ob.model", dir))
	defer ob.Close()
	if err == nil {
		oBiases.UnmarshalBinaryFrom(ob)
	}

	r, _ := hWeights.Dims()
	_, c := oWeights.Dims()

	model := NewModel(r, c)
	model.hiddenLayer.weights = &hWeights
	model.hiddenLayer.biases = &hBiases
	model.outputLayer.weights = &oWeights
	model.outputLayer.biases = &oBiases

	return model
}

func (m *Model) Forward(input *mat.Dense) *mat.Dense {
	m.hiddenLayer.Forward(input)
	m.reluLayer.Forward(m.hiddenLayer.output)
	m.outputLayer.Forward(m.reluLayer.output)
	m.softmaxLayer.Forward(m.outputLayer.output)
	return m.softmaxLayer.output
}

func (m *Model) Train(dl DataLoader, epochs int) {
	batchSize := 10
	criterion := NewCrossEntropy()
	optimizer := NewAdam(0.001)

	for epoch := 1; epoch <= epochs; epoch++ {
		runningLoss := 0.0
		runningAccuracy := 0.0
		n_batches := dl.Size() / batchSize

		for i := 0; i < n_batches; i++ {
			input, output := dl.Sample(batchSize*i, batchSize)

			// 1. Forward
			outputPred := m.Forward(input)

			// Calculate metrics
			runningLoss += criterion.Forward(outputPred, output)
			runningAccuracy += Accuracy(outputPred, output)

			// 2. Backward Propagation

			// Derivative of loss function respect to softmax input, that means:
			// dinputs => dloss/dz = dloss/dsoftmax * dsoftmax/dz
			dinputs := num.Sub(outputPred, output)

			m.outputLayer.Backward(dinputs)
			m.reluLayer.Backward(m.outputLayer.dinputs)
			m.hiddenLayer.Backward(m.reluLayer.dinputs)

			// 3. Update Parameters
			optimizer.PreUpdateParams()
			optimizer.UpdateParams(m.outputLayer)
			optimizer.UpdateParams(m.hiddenLayer)
			optimizer.PostUpdateParams()
		}

		fmt.Printf(
			"Epoch: %v/%v, loss: %.4f, acc: %.4f\n",
			epoch, epochs, runningLoss/float64(n_batches), runningAccuracy/float64(n_batches),
		)
	}
}

func (m *Model) Save(dir string) {
	hw, err := os.Create(fmt.Sprintf("%s/hw.model", dir))
	defer hw.Close()
	if err == nil {
		m.hiddenLayer.weights.MarshalBinaryTo(hw)
	}

	ow, err := os.Create(fmt.Sprintf("%s/ow.model", dir))
	defer ow.Close()
	if err == nil {
		m.outputLayer.weights.MarshalBinaryTo(ow)
	}

	hb, err := os.Create(fmt.Sprintf("%s/hb.model", dir))
	defer hb.Close()
	if err == nil {
		m.hiddenLayer.biases.MarshalBinaryTo(hb)
	}

	ob, err := os.Create(fmt.Sprintf("%s/ob.model", dir))
	defer ob.Close()
	if err == nil {
		m.outputLayer.biases.MarshalBinaryTo(ob)
	}
}
