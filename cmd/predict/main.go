package main

import (
	"bufio"
	"fmt"
	"os"

	"github.com/arturobermejo/gobot/neunet"
)

func main() {
	model := neunet.LoadModel("output/hw.model", "output/ow.model", "output/hb.model", "output/ob.model")
	inVocab := neunet.LoadVocab("output/invocab.model")
	outVocab := neunet.LoadVocab("output/outvocab.model")

	fmt.Println("Hello!")
	reader := bufio.NewReader(os.Stdin)
	msg, _ := reader.ReadString('\n')

	input := neunet.OneHotEncode([]string{msg}, inVocab)
	output := model.Forward(input)
	intent := neunet.OneHotDecode(output, outVocab)

	fmt.Println(intent)
}
