package main

import (
	"github.com/arturobermejo/gobot/neunet"
)

func main() {
	dl := neunet.NewDataLoader("data/train")
	dl.Save()

	model := neunet.NewModel(len(dl.InVocab()), len(dl.OutVocab()))
	model.Train(dl, 200)
	model.Save("output")
}
