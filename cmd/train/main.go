package main

import (
	"bufio"
	"log"
	"os"
	"strings"

	"github.com/arturobermejo/gobot/neunet"
	"gonum.org/v1/gonum/mat"
)

func main() {
	f, err := os.Open("data/train")

	if err != nil {
		log.Fatal(err)
	}

	defer f.Close()

	s := bufio.NewScanner(f)

	inVocab := neunet.NewVocab()
	outVocab := neunet.NewVocab()

	inData := []string{}
	outData := []string{}

	for s.Scan() {
		sample := strings.Split(s.Text(), "|")
		inSample := neunet.CleanText(sample[0])
		inVocab.ProcessText(inSample)
		outVocab.ProcessText(sample[1])
		inData = append(inData, neunet.CleanText(inSample))
		outData = append(outData, sample[1])
	}

	inVocab.Save("output/invocab.model")
	outVocab.Save("output/outvocab.model")

	inputs := mat.NewDense(len(inData), inVocab.Size(), nil)
	outputs := mat.NewDense(1, len(outData), nil)

	for i := 0; i < len(inData); i++ {
		inVocab.OneHotEncode(inData[i], inputs, i, 0)
		outVocab.IndexEncode(outData[i], outputs, i)
	}

	dl := neunet.NewDataLoader(inputs, outputs, 10)

	model := neunet.NewModel(inVocab.Size(), outVocab.Size())
	model.Train(dl, 200)
	model.Save("output")
}
