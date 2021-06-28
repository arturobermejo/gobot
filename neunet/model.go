package neunet

import (
	"fmt"
	"os"

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
	hiddenSize := 16

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

func LoadModel(hwpath, owpath, hbpath, obpath string) *Model {
	var hWeights, oWeights, hBiases, oBiases mat.Dense

	hw, err := os.Open(hwpath)
	defer hw.Close()
	if err == nil {
		hWeights.UnmarshalBinaryFrom(hw)
	}

	ow, err := os.Open(owpath)
	defer ow.Close()
	if err == nil {
		oWeights.UnmarshalBinaryFrom(ow)
	}

	hb, err := os.Open(hbpath)
	defer hb.Close()
	if err == nil {
		hBiases.UnmarshalBinaryFrom(hb)
	}

	ob, err := os.Open(obpath)
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
			outputPred := m.Forward(input)

			loss := criterion.Forward(outputPred, output)
			runningAccuracy += accuracy(outputPred, output)

			runningLoss += loss

			// Backward Propagation

			// Derivative of loss function respect to softmax input, that means:
			// dinputs => dloss/dz = dloss/dsoftmax * dsoftmax/dz
			dinputs := matrixSubtract(outputPred, output)

			m.outputLayer.Backward(dinputs)
			m.reluLayer.Backward(m.outputLayer.dinputs)
			m.hiddenLayer.Backward(m.reluLayer.dinputs)

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

func (m *Model) Save() {
	hw, err := os.Create("output/hw.model")
	defer hw.Close()
	if err == nil {
		m.hiddenLayer.weights.MarshalBinaryTo(hw)
	}

	ow, err := os.Create("output/ow.model")
	defer ow.Close()
	if err == nil {
		m.outputLayer.weights.MarshalBinaryTo(ow)
	}

	hb, err := os.Create("output/hb.model")
	defer hb.Close()
	if err == nil {
		m.hiddenLayer.biases.MarshalBinaryTo(hb)
	}

	ob, err := os.Create("output/ob.model")
	defer ob.Close()
	if err == nil {
		m.outputLayer.biases.MarshalBinaryTo(ob)
	}
}
