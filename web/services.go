package web

import (
	"math/rand"

	"github.com/arturobermejo/gobot/neunet"
)

func ChatService(msg string) (string, string) {
	model := neunet.LoadModel("output")
	inVocab := neunet.LoadVocab("output/invocab.model")
	outVocab := neunet.LoadVocab("output/outvocab.model")

	input := neunet.OneHotEncode([]string{msg}, inVocab)
	output := model.Forward(input)
	intent := neunet.OneHotDecode(output, outVocab)

	responses := ResponseSet[intent]

	return intent, responses[rand.Intn(len(responses))]
}
