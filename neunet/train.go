package neunet

import (
	"fmt"
)

func Train() {
	dl := NewDataLoader()
	dl.FromFile("data/intent_train")

	inputData, outputData := dl.Data()

	voca := NewVocabulary(inputData, outputData)

	model := NewModel(voca.GetInputSize(), 16, voca.GetOutputSize())

	epochs := 50
	batchSize := 25

	criterion := NewCrossEntropy()
	optimizer := NewSGD(0.001)

	for epoch := 1; epoch <= epochs; epoch++ {
		for i := 0; i < len(inputData)/batchSize; i++ {
			inputData, outputData := dl.Sample(batchSize*i, batchSize)

			input := voca.GetInputMatrix(inputData)

			output := voca.GetOutputMatrix(outputData)
			outputPred := model.Forward(input)

			loss := criterion.Forward(outputPred, output)
			accuracy := accuracy(outputPred, output)

			if epoch%1 == 0 {
				fmt.Printf("epoch: %v, acc: %v, loss: %v\n", epoch, accuracy, loss)
			}

			// Backward Propagation

			// Derivative of loss function respect to softmax input, that means:
			// dinputs => dloss/dz = dloss/dsoftmax * dsoftmax/dz
			dinputs := matrixSubtract(outputPred, output)

			model.outputLayer.Backward(dinputs)
			model.reluLayer.Backward(model.outputLayer.dinputs)
			model.hiddenLayer.Backward(model.reluLayer.dinputs)

			optimizer.UpdateParameters(model.outputLayer)
			optimizer.UpdateParameters(model.hiddenLayer)
		}
	}

	model.Save()
}
