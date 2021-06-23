package neunet

func Train() {
	dl := NewDataLoader()
	inputData, outputData := dl.FromFile("data/messages", "data/categories")

	voca := NewVocabulary(inputData, outputData)
	model := NewModel(voca.GetInputSize(), 10, 1)
	learningRate := 0.001

	n := len(inputData)
	for i := 0; i < n; i++ {
		message := voca.GetInputVector(inputData[i])
		category := voca.GetOutputVector(outputData[i])

		outputEst := model.Forward(message) // estimated category

		//Output error
		outputError := matrixSubtract(outputEst, category)
		outputDelta := matrixMultiply(outputError, SigmoidOutputDerivative(outputEst)) // why?

		// Backpropagated error
		hiddelLayerError := matrixDot(outputDelta, model.weights2.T())
		hiddenLayerDelta := hiddelLayerError

		//Update weights
		model.weights2 = matrixSubtract(
			model.weights2,
			matrixScale(learningRate, matrixDot(model.hiddenLayer.T(), outputDelta)))

		model.weights1 = matrixSubtract(
			model.weights1,
			matrixScale(learningRate, matrixDot(message.T(), hiddenLayerDelta)))

		// fmt.Println(outputEst.At(0, 0), category.At(0, 0))
	}
}
