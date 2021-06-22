package neunet

import (
	"io/ioutil"
	"log"
	"strings"
)

type DataLoader struct {
}

func NewDataLoader() *DataLoader {
	return &DataLoader{}
}

// Load training data from files, note this load all the data in memory
func (dl *DataLoader) FromFile(inputPath string, outputPath string) ([]string, []string) {
	input_data, err := ioutil.ReadFile(inputPath)

	if err != nil {
		log.Fatal(err)
	}

	output_data, err := ioutil.ReadFile(outputPath)

	if err != nil {
		log.Fatal(err)
	}

	input := strings.Split(string(input_data), "\n")
	output := strings.Split(string(output_data), "\n")

	return input, output
}
