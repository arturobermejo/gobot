package neunet

import (
	"fmt"

	"gonum.org/v1/gonum/stat"
)

func Train() {
	dl := NewDataLoader()
	inputData, outputData := dl.FromFile("data/messages", "data/categories")

	voca := NewVocabulary(inputData, outputData)
	model := NewModel(voca.GetInputSize(), 4, voca.GetOutputSize())
	learningRate := 0.001

	n := 2 //len(inputData)
	for i := 0; i < n; i++ {
		input := voca.GetInputVector(inputData[i])
		category := voca.GetOutputVector(outputData[i])

		outputEst := model.Forward(input) // estimated category

		fmt.Println(voca)

		loss := stat.CrossEntropy(category.RawMatrix().Data, outputEst.RawMatrix().Data)

		dz2 := matrixSubtract(outputEst, category)   // 10x1
		dw2 := matrixDot(model.hiddenLayer.T(), dz2) // 10x7

		dz1 := matrixDot(dz2, model.weightsOutput.T())
		dw1 := matrixDot(input.T(), dz1) // 35x10

		model.weightsOutput = matrixSubtract(model.weightsOutput, matrixScale(learningRate, dw2))
		model.weightsHidden = matrixSubtract(model.weightsHidden, matrixScale(learningRate, dw1))

		fmt.Println(loss)
	}

	model.Save()
}
