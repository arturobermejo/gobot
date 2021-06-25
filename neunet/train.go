package neunet

import (
	"fmt"
)

func Train() {
	dl := NewDataLoader()
	inputData, outputData := dl.FromFile("data/messages", "data/categories")

	voca := NewVocabulary(inputData, outputData)
	model := NewModel(voca.GetInputSize(), 4, voca.GetOutputSize())
	learningRate := 0.001
	epochs := 1000

	criterion := NewCrossEntropy()

	for epoch := 1; epoch <= epochs; epoch++ {
		input := voca.GetInputMatrix(inputData)
		output := voca.GetOutputMatrix(outputData)
		outputPred := model.Forward(input)

		loss := criterion.Fordward(output, outputPred)

		dz2 := matrixSubtract(outputPred, output)
		dw2 := matrixDot(model.hiddenLayer.output.T(), dz2)

		dz1 := matrixDot(dz2, model.outputLayer.weights.T())
		dw1 := matrixDot(input.T(), dz1)

		model.outputLayer.weights = matrixSubtract(model.outputLayer.weights, matrixScale(learningRate, dw2))
		model.hiddenLayer.weights = matrixSubtract(model.hiddenLayer.weights, matrixScale(learningRate, dw1))

		if epoch%10 == 0 {
			fmt.Printf("epoch: %v, loss: %v\n", epoch, loss)
		}
	}

	model.Save()
}
