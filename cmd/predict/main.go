package main

import (
	"bufio"
	"fmt"
	"os"

	"github.com/arturobermejo/gobot/neunet"
	"gonum.org/v1/gonum/mat"
)

func main() {
	model := neunet.LoadModel("output")
	inVocab := neunet.LoadVocab("output/invocab.model")
	outVocab := neunet.LoadVocab("output/outvocab.model")

	fmt.Println("Hello!")
	reader := bufio.NewReader(os.Stdin)
	msg, _ := reader.ReadString('\n')

	input := mat.NewDense(1, inVocab.Size(), nil)
	inVocab.OneHotEncode(neunet.CleanText(msg), input, 0, 2)

	var intent string
	var prob float64

	if mat.Sum(input) == 0 {
		intent = ""
		prob = 0
	} else {
		output := neunet.Softmax(model.Forward(input))
		intent, prob = outVocab.OutputDecode(output)
	}

	fmt.Println(intent, prob)
}
