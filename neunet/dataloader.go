package neunet

import (
	"encoding/gob"
	"io/ioutil"
	"log"
	"os"
	"strings"

	"gonum.org/v1/gonum/mat"
)

type dataLoader struct {
	data     []string
	inputs   *mat.Dense
	outputs  *mat.Dense
	inVocab  map[string]int
	outVocab map[string]int
}

func NewDataLoader(path string) *dataLoader {
	dl := &dataLoader{}
	dl.fromFile(path)
	dl.parseData()
	return dl
}

func (dl *dataLoader) fromFile(path string) {
	f, err := ioutil.ReadFile(path)

	if err != nil {
		log.Fatal(err)
	}

	dl.data = strings.Split(string(f), "\n")
}

func (dl *dataLoader) parseData() {

	txtInputs := []string{}
	txtOutputs := []string{}

	for _, line := range dl.data {
		sample := strings.Split(line, "|")
		txtInputs = append(txtInputs, strings.ToLower(sample[0]))
		txtOutputs = append(txtOutputs, sample[1])
	}

	dl.inVocab = GetVocab(txtInputs)
	dl.outVocab = GetVocab(txtOutputs)

	dl.inputs = OneHotEncode(txtInputs, dl.inVocab)
	dl.outputs = OneHotEncode(txtOutputs, dl.outVocab)
}

func (dl *dataLoader) Sample(i, offset int) (*mat.Dense, *mat.Dense) {
	r, _ := dl.inputs.Dims()

	e := i + offset

	if e > r {
		e = r
	}

	inputs := dl.inputs.Slice(i, e, 0, len(dl.inVocab))
	outputs := dl.outputs.Slice(i, e, 0, len(dl.outVocab))

	return inputs.(*mat.Dense), outputs.(*mat.Dense)
}

func (dl *dataLoader) Size() int {
	return len(dl.data)
}

func (dl *dataLoader) InVocab() map[string]int {
	return dl.inVocab
}

func (dl *dataLoader) OutVocab() map[string]int {
	return dl.outVocab
}

func (dl *dataLoader) Save() {
	iv, err := os.Create("output/invocab.model")
	defer iv.Close()
	if err == nil {
		e := gob.NewEncoder(iv)
		e.Encode(dl.inVocab)
	}

	ov, err := os.Create("output/outvocab.model")
	defer ov.Close()
	if err == nil {
		e := gob.NewEncoder(ov)
		e.Encode(dl.outVocab)
	}
}

func GetVocab(data []string) map[string]int {
	counter := map[string]int{}

	i := 0

	for _, line := range data {
		for _, word := range strings.Split(line, " ") {

			_, ok := counter[word]

			if !ok {
				counter[word] = i
				i += 1
			}
		}
	}

	return counter
}

func OneHotEncode(data []string, vocab map[string]int) *mat.Dense {
	c := mat.NewDense(len(data), len(vocab), nil)

	for i, line := range data {
		for _, word := range strings.Split(strings.TrimSuffix(line, "\n"), " ") {
			idx, ok := vocab[word]

			if ok {
				c.Set(i, idx, 1)
			}
		}
	}

	return c
}

func OneHotDecode(data *mat.Dense, vocab map[string]int) string {
	idx := Argmax(data.RawMatrix().Data)

	for k, v := range vocab {
		if v == idx {
			return k
		}
	}

	return ""
}

func LoadVocab(path string) map[string]int {
	var data map[string]int

	f, err := os.Open(path)
	defer f.Close()

	if err != nil {
		log.Fatal(err)
	}

	D := gob.NewDecoder(f)
	err = D.Decode(&data)

	if err != nil {
		log.Fatal(err)
	}

	return data
}
