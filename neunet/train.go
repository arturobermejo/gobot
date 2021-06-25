package neunet

import (
	"fmt"
)

func Train() {
	dl := NewDataLoader()
	inputData, outputData := dl.FromFile("data/messages", "data/categories")

	voca := NewVocabulary(inputData, outputData)
	model := NewModel(voca.GetInputSize(), 4, voca.GetOutputSize())
	learningRate := 0.000001
	epochs := 100

	criterion := NewCrossEntropy()

	for epoch := 1; epoch <= epochs; epoch++ {
		input := voca.GetInputMatrix(inputData)

		// Calculate loss
		output := voca.GetOutputMatrix(outputData)
		outputPred := model.Forward(input)
		loss := criterion.Fordward(outputPred, output)

		// Derivative of loss function respect to softmax input, that means:
		// dinputs => dloss/dz = dloss/dsoftmax * dsoftmax/dz
		dinputs := matrixSubtract(outputPred, output)

		// output layer
		model.outputLayer.Backward(dinputs)
		model.reluLayer.Backward(model.outputLayer.dinputs)
		model.hiddenLayer.Backward(model.reluLayer.dinputs)

		model.outputLayer.weights = matrixSubtract(
			model.outputLayer.weights, matrixScale(learningRate, model.outputLayer.dweights))
		model.hiddenLayer.weights = matrixSubtract(
			model.hiddenLayer.weights, matrixScale(learningRate, model.hiddenLayer.dweights))

		if epoch%10 == 0 {
			fmt.Printf("epoch: %v, loss: %v\n", epoch, loss)
		}
	}

	model.Save()
}
