package main

import (
	"bufio"
	"fmt"
	"os"

	"github.com/arturobermejo/gobot/neunet"
)

func main() {
	dl := neunet.NewDataLoader()
	inputData, outputData := dl.FromFile("data/messages", "data/categories")
	voca := neunet.NewVocabulary(inputData, outputData)
	model := neunet.NewModel(voca.GetInputSize(), 4, voca.GetOutputSize())
	model.Load()

	reader := bufio.NewReader(os.Stdin)

	msg, _ := reader.ReadString('\n')

	message := voca.GetInputMatrix([]string{msg})
	result := model.Forward(message)

	fmt.Println(result)
}
