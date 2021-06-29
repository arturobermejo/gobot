package main

import (
	"bufio"
	"fmt"
	"os"

	"github.com/arturobermejo/gobot/neunet"
)

func main() {
	model := neunet.LoadModel("output")
	inVocab := neunet.LoadVocab("output/invocab.model")
	outVocab := neunet.LoadVocab("output/outvocab.model")

	fmt.Println("Hello!")
	reader := bufio.NewReader(os.Stdin)
	msg, _ := reader.ReadString('\n')

	input := neunet.OneHotEncode([]string{msg}, inVocab)
	output := model.Forward(input)
	intent, prob := neunet.OutputDecode(output, outVocab)

	fmt.Println(intent, prob)
}
