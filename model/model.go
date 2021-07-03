package model

import (
	"github.com/arturobermejo/gobot/ml/fn"
	"github.com/arturobermejo/gobot/ml/nn"
)

type Model struct {
	inputSize  int
	hiddenSize int
	outputSize int

	hiddenLayer *nn.Linear
	reluLayer   *nn.ReLU
	outputLayer *nn.Linear
}

func NewModel(inputSize, outputSize int) *Model {
	hiddenSize := 32

	return &Model{
		inputSize:   inputSize,
		hiddenSize:  hiddenSize,
		outputSize:  outputSize,
		hiddenLayer: nn.NewLinear(inputSize, hiddenSize),
		reluLayer:   nn.NewReLU(),
		outputLayer: nn.NewLinear(hiddenSize, outputSize),
	}
}

func (m *Model) Forward(input *fn.Tensor) *fn.Tensor {
	m.hiddenLayer.Forward(input)
	m.reluLayer.Forward(m.hiddenLayer.Out)
	m.outputLayer.Forward(m.reluLayer.Out)
	return m.outputLayer.Out
}
