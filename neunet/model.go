package neunet

import (
	"os"

	"gonum.org/v1/gonum/mat"
)

type Model struct {
	inputSize  int
	hiddenSize int
	outputSize int

	hiddenLayer   *mat.Dense
	outputLayer   *mat.Dense
	weightsHidden *mat.Dense
	weightsOutput *mat.Dense
}

func NewModel(inputSize, hiddenSize, outputSize int) *Model {
	return &Model{
		inputSize:     inputSize,
		hiddenSize:    hiddenSize,
		outputSize:    outputSize,
		weightsHidden: mat.NewDense(inputSize, hiddenSize, randomArray(inputSize*hiddenSize, float64(inputSize))),
		weightsOutput: mat.NewDense(hiddenSize, outputSize, randomArray(hiddenSize*outputSize, float64(hiddenSize))),
	}
}

// Z[L] = W[L]xA[L-1]
func (m *Model) Forward(input *mat.Dense) *mat.Dense {
	m.hiddenLayer = matrixDot(input, m.weightsHidden)
	m.outputLayer = matrixDot(m.hiddenLayer, m.weightsOutput)
	return softmax(m.outputLayer)
}

func (m *Model) Save() {
	h, err := os.Create("data/weights1.model")
	defer h.Close()
	if err == nil {
		m.weightsHidden.MarshalBinaryTo(h)
	}
	o, err := os.Create("data/weights2.model")
	defer o.Close()
	if err == nil {
		m.weightsOutput.MarshalBinaryTo(o)
	}
}

func (m *Model) Load() {
	h, err := os.Open("data/weightsHidden.model")
	defer h.Close()
	if err == nil {
		m.weightsHidden.Reset()
		m.weightsHidden.UnmarshalBinaryFrom(h)
	}
	o, err := os.Open("data/weightsOutput.model")
	defer o.Close()
	if err == nil {
		m.weightsOutput.Reset()
		m.weightsOutput.UnmarshalBinaryFrom(o)
	}
}

// TODO: REMOVE
func (m *Model) GetWeight() *mat.Dense {
	return m.weightsHidden
}
