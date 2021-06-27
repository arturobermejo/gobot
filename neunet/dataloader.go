package neunet

import (
	"io/ioutil"
	"log"
	"strings"
)

type DataLoader struct {
	data []string
}

func NewDataLoader() *DataLoader {
	return &DataLoader{}
}

// Load training data from files, note this load all the data in memory
func (dl *DataLoader) FromFile(path string) {
	f, err := ioutil.ReadFile(path)

	if err != nil {
		log.Fatal(err)
	}

	dl.data = strings.Split(string(f), "\n")
}

func (dl *DataLoader) Data() ([]string, []string) {
	return dl.split(dl.data)
}

func (dl *DataLoader) Sample(s, offset int) ([]string, []string) {
	l := len(dl.data)

	e := s + offset

	if e > l {
		e = l
	}

	return dl.split(dl.data[s:e])
}

func (dl *DataLoader) split(data []string) ([]string, []string) {
	var inputs, outputs []string

	for _, v := range data {
		sample := strings.Split(v, "|")
		inputs = append(inputs, sample[0])
		outputs = append(outputs, sample[1])
	}

	return inputs, outputs
}
