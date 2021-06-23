package main

import (
	"fmt"

	"github.com/arturobermejo/gobot/neunet"
)

func main() {
	dl := neunet.NewDataLoader()
	inputData, outputData := dl.FromFile("data/messages", "data/categories")
	voca := neunet.NewVocabulary(inputData, outputData)
	model := neunet.NewModel(voca.GetInputSize(), 10, 1)
	model.Load()

	message := voca.GetInputVector("hola arturo")
	result := model.Forward(message)

	fmt.Println(result)
}
