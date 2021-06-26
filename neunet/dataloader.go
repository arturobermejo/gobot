package neunet

import (
	"io/ioutil"
	"log"
	"math/rand"
	"strings"
	"time"
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

func (dl *DataLoader) Sample(n int) ([]string, []string) {
	rand.Seed(time.Now().UnixNano())
	rand.Shuffle(len(dl.data), func(i, j int) { dl.data[i], dl.data[j] = dl.data[j], dl.data[i] })
	return dl.split(dl.data[:n])
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
