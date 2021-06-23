package neunet

import (
	"os"

	"gonum.org/v1/gonum/mat"
)

type Model struct {
	inputSize  int
	hiddenSize int
	outputSize int

	hiddenLayer *mat.Dense
	outputLayer *mat.Dense
	weights1    *mat.Dense
	weights2    *mat.Dense
}

func NewModel(inputSize, hiddenSize, outputSize int) *Model {
	return &Model{
		inputSize:  inputSize,
		hiddenSize: hiddenSize,
		outputSize: outputSize,
		weights1:   mat.NewDense(inputSize, hiddenSize, nil), // randomarray?
		weights2:   mat.NewDense(hiddenSize, outputSize, randomArray(hiddenSize*outputSize, float64(hiddenSize))),
	}
}

func (m *Model) Forward(input *mat.Dense) *mat.Dense {
	m.hiddenLayer = matrixDot(input, m.weights1)
	m.outputLayer = sigmoid(matrixDot(m.hiddenLayer, m.weights2))
	return m.outputLayer
}

func (m *Model) Save() {
	h, err := os.Create("data/weights1.model")
	defer h.Close()
	if err == nil {
		m.weights1.MarshalBinaryTo(h)
	}
	o, err := os.Create("data/weights2.model")
	defer o.Close()
	if err == nil {
		m.weights2.MarshalBinaryTo(o)
	}
}

func (m *Model) Load() {
	h, err := os.Open("data/weights1.model")
	defer h.Close()
	if err == nil {
		m.weights1.Reset()
		m.weights1.UnmarshalBinaryFrom(h)
	}
	o, err := os.Open("data/weights2.model")
	defer o.Close()
	if err == nil {
		m.weights2.Reset()
		m.weights2.UnmarshalBinaryFrom(o)
	}
}
