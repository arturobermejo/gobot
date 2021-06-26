package neunet

import (
	"fmt"
)

func Train() {
	dl := NewDataLoader()
	inputData, outputData := dl.FromFile("data/messages", "data/categories")

	voca := NewVocabulary(inputData, outputData)
	model := NewModel(voca.GetInputSize(), 10, voca.GetOutputSize())
	learningRate := 0.000001
	epochs := 100

	criterion := NewCrossEntropy()
	optimizer := NewSGD(learningRate)

	for epoch := 1; epoch <= epochs; epoch++ {
		input := voca.GetInputMatrix(inputData)

		// Calculate loss
		output := voca.GetOutputMatrix(outputData)
		outputPred := model.Forward(input)
		loss := criterion.Forward(outputPred, output)

		accuracy := accuracy(outputPred, output)

		// Backward Propagation

		// Derivative of loss function respect to softmax input, that means:
		// dinputs => dloss/dz = dloss/dsoftmax * dsoftmax/dz
		dinputs := matrixSubtract(outputPred, output)

		model.outputLayer.Backward(dinputs)
		model.reluLayer.Backward(model.outputLayer.dinputs)
		model.hiddenLayer.Backward(model.reluLayer.dinputs)

		optimizer.UpdateParameters(model.outputLayer)
		optimizer.UpdateParameters(model.hiddenLayer)

		if epoch%10 == 0 {
			fmt.Printf("epoch: %v, acc: %v, loss: %v\n", epoch, accuracy, loss)
		}
	}

	model.Save()
}
