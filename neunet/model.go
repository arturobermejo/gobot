package neunet

import (
	"os"

	"gonum.org/v1/gonum/mat"
)

type Model struct {
	inputSize  int
	hiddenSize int
	outputSize int

	hiddenLayer  *LinearLayer
	reluLayer    *ReLUActivation
	outputLayer  *LinearLayer
	softmaxLayer *SoftmaxActivation
}

func NewModel(inputSize, hiddenSize, outputSize int) *Model {
	return &Model{
		inputSize:    inputSize,
		hiddenSize:   hiddenSize,
		outputSize:   outputSize,
		hiddenLayer:  NewLinearLayer(inputSize, hiddenSize),
		reluLayer:    NewReLUActivation(),
		outputLayer:  NewLinearLayer(hiddenSize, outputSize),
		softmaxLayer: NewSoftmaxActivation(),
	}
}

func (m *Model) Forward(input *mat.Dense) *mat.Dense {
	m.hiddenLayer.Fordward(input)
	m.reluLayer.Fordward(m.hiddenLayer.output)
	m.outputLayer.Fordward(m.reluLayer.output)
	m.softmaxLayer.Fordward(m.outputLayer.output)
	return m.softmaxLayer.output
}

func (m *Model) Save() {
	h, err := os.Create("data/weightsHidden.model")
	defer h.Close()
	if err == nil {
		m.hiddenLayer.weights.MarshalBinaryTo(h)
	}
	o, err := os.Create("data/weightsOutput.model")
	defer o.Close()
	if err == nil {
		m.outputLayer.weights.MarshalBinaryTo(o)
	}
}

func (m *Model) Load() {
	h, err := os.Open("data/weightsHidden.model")
	defer h.Close()
	if err == nil {
		m.hiddenLayer.weights.Reset()
		m.hiddenLayer.weights.UnmarshalBinaryFrom(h)
	}
	o, err := os.Open("data/weightsOutput.model")
	defer o.Close()
	if err == nil {
		m.outputLayer.weights.Reset()
		m.outputLayer.weights.UnmarshalBinaryFrom(o)
	}
}
