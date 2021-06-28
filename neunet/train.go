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
	optimizer := NewSGD(0.001, 0.0, 0.0)
	// optimizer := NewAdam(0.001)

	for epoch := 1; epoch <= epochs; epoch++ {

		runningLoss := 0.0
		runningAccuracy := 0.0
		n_batches := len(inputData) / batchSize
		// n_batches := 1

		for i := 0; i < n_batches; i++ {
			inputData, outputData := dl.Sample(batchSize*i, batchSize)

			input := voca.GetInputMatrix(inputData)

			output := voca.GetOutputMatrix(outputData)
			outputPred := model.Forward(input)

			loss := criterion.Forward(outputPred, output)
			runningAccuracy += accuracy(outputPred, output)

			runningLoss += loss

			// Backward Propagation

			// Derivative of loss function respect to softmax input, that means:
			// dinputs => dloss/dz = dloss/dsoftmax * dsoftmax/dz
			dinputs := matrixSubtract(outputPred, output)

			model.outputLayer.Backward(dinputs)
			model.reluLayer.Backward(model.outputLayer.dinputs)
			model.hiddenLayer.Backward(model.reluLayer.dinputs)

			optimizer.PreUpdateParams()
			optimizer.UpdateParams(model.outputLayer)
			optimizer.UpdateParams(model.hiddenLayer)
			optimizer.PostUpdateParams()
		}

		fmt.Printf(
			"Epoch: %v/%v, loss: %.4f, acc: %.4f\n",
			epoch, epochs, runningLoss/float64(n_batches), runningAccuracy/float64(n_batches),
		)
	}

	model.Save()
}
