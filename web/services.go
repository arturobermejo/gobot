package web

import (
	"math/rand"

	"github.com/arturobermejo/gobot/neunet"
	"gonum.org/v1/gonum/mat"
)

func ChatService(msg string) (intent string, prob float64, res string) {
	model := neunet.LoadModel("output")
	inVocab := neunet.LoadVocab("output/invocab.model", neunet.CleanText)
	outVocab := neunet.LoadVocab("output/outvocab.model", nil)

	input := mat.NewDense(1, inVocab.Size(), nil)
	inVocab.OneHotEncode(msg, input, 0, 2)

	if mat.Sum(input) == 0 {
		intent = ""
		prob = 0.0
	} else {
		output := neunet.Softmax(model.Forward(input))
		intent, prob = outVocab.OutputDecode(output)
	}

	if prob > 0.5 {
		responses := ResponseSet[intent]
		res = responses[rand.Intn(len(responses))]
	} else {
		res = "Disculpa, no entendimos tu mensaje"
	}

	return
}
