package neunet

import (
	"encoding/gob"
	"fmt"
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
		txtInputs = append(txtInputs, CleanText(sample[0]))
		txtOutputs = append(txtOutputs, sample[1])
	}

	dl.inVocab = GetVocab(txtInputs)
	dl.outVocab = GetVocab(txtOutputs)

	dl.inputs = EncodeInputs(txtInputs, dl.inVocab, 0)
	dl.outputs = EncodeOutputs(txtOutputs, dl.outVocab)
}

func (dl *dataLoader) Sample(i, offset int) (*mat.Dense, *mat.Dense) {
	r, _ := dl.inputs.Dims()

	e := i + offset

	if e > r {
		e = r
	}

	inputs := dl.inputs.Slice(i, e, 0, len(dl.inVocab))
	outputs := dl.outputs.Slice(0, 1, i, e)

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

func (dl *dataLoader) Save(dir string) {
	iv, err := os.Create(fmt.Sprintf("%s/invocab.model", dir))
	defer iv.Close()
	if err == nil {
		e := gob.NewEncoder(iv)
		e.Encode(dl.inVocab)
	}

	ov, err := os.Create(fmt.Sprintf("%s/outvocab.model", dir))
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

func EncodeInputs(data []string, vocab map[string]int, threshold int) *mat.Dense {
	c := mat.NewDense(len(data), len(vocab), nil)

	for i, line := range data {
		words := strings.Split(CleanText(line), " ")

		for _, word := range words {
			idx, ok := vocab[word]

			if ok {
				c.Set(i, idx, 1)
			} else {
				for k, v := range vocab {
					d := Levenshtein(&k, &word)
					if d < threshold {
						c.Set(i, v, 1)
						break
					}
				}
			}
		}
	}

	return c
}

func EncodeOutputs(data []string, vocab map[string]int) *mat.Dense {
	o := mat.NewDense(1, len(data), nil)

	for i, output := range data {
		o.Set(0, i, float64(vocab[output]))
	}

	return o
}

func OutputDecode(data *mat.Dense, vocab map[string]int) (string, float64) {
	idx, prob := Argmax(data.RawMatrix().Data)

	for k, v := range vocab {
		if v == idx {
			return k, prob
		}
	}

	return "", 0.0
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

func Levenshtein(a, b *string) int {
	la := len(*a)
	lb := len(*b)
	d := make([]int, la+1)
	var lastdiag, olddiag, temp int

	for i := 1; i <= la; i++ {
		d[i] = i
	}
	for i := 1; i <= lb; i++ {
		d[0] = i
		lastdiag = i - 1
		for j := 1; j <= la; j++ {
			olddiag = d[j]
			min := d[j] + 1
			if (d[j-1] + 1) < min {
				min = d[j-1] + 1
			}
			if (*a)[j-1] == (*b)[i-1] {
				temp = 0
			} else {
				temp = 1
			}
			if (lastdiag + temp) < min {
				min = lastdiag + temp
			}
			d[j] = min
			lastdiag = olddiag
		}
	}
	return d[la]
}
