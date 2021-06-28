package main

import (
	"bufio"
	"fmt"
	"os"

	"github.com/arturobermejo/gobot/neunet"
)

func main() {
	dl := neunet.NewDataLoader()
	dl.FromFile("data/intent_train")
	inputData, outputData := dl.Data()
	voca := neunet.NewVocabulary(inputData, outputData)
	model := neunet.NewModel(voca.GetInputSize(), 16, voca.GetOutputSize())
	model.Load()

	reader := bufio.NewReader(os.Stdin)

	msg, _ := reader.ReadString('\n')

	message := voca.GetInputMatrix([]string{msg})
	result := model.Forward(message)
	fmt.Println(voca.GetOutput(neunet.Argmax(result.RawMatrix().Data)))
}
