package web

import (
	"math/rand"

	"github.com/arturobermejo/gobot/neunet"
)

func ChatService(msg string) (string, float64, string) {
	model := neunet.LoadModel("output")
	inVocab := neunet.LoadVocab("output/invocab.model")
	outVocab := neunet.LoadVocab("output/outvocab.model")

	input := neunet.OneHotEncode([]string{msg}, inVocab, 3)
	output := model.Forward(input)
	intent, prob := neunet.OutputDecode(output, outVocab)

	var response string

	if prob > 0.3 {
		responses := ResponseSet[intent]
		response = responses[rand.Intn(len(responses))]
	} else {
		response = "Disculpa, no entendimos tu mensaje"
	}

	return intent, prob, response
}
