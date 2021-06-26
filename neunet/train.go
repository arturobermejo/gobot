package neunet

import (
	"fmt"
)

func Train() {
	dl := NewDataLoader()
	dl.FromFile("data/train")

	inputData, outputData := dl.Data()

	voca := NewVocabulary(inputData, outputData)

	model := NewModel(voca.GetInputSize(), 10, voca.GetOutputSize())

	learningRate := 0.001
	epochs := 500

	criterion := NewCrossEntropy()
	optimizer := NewSGD(learningRate)

	for epoch := 1; epoch <= epochs; epoch++ {
		inputData, outputData := dl.Sample(10)

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
