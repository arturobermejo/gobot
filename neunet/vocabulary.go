package neunet

import (
	"strings"

	"gonum.org/v1/gonum/mat"
)

type Vocabulary struct {
	inputVocabulary  map[string]int
	outputVocabulary map[string]int
}

func NewVocabulary(inputs, outputs []string) *Vocabulary {
	v := Vocabulary{
		inputVocabulary:  getVocabulary(inputs),
		outputVocabulary: getVocabulary(outputs),
	}
	return &v
}

func (v *Vocabulary) GetInputSize() int {
	return len(v.inputVocabulary)
}

func (v *Vocabulary) GetOutputSize() int {
	return len(v.outputVocabulary)
}

func (v *Vocabulary) GetOutputMatrix(outputs []string) *mat.Dense {
	c := mat.NewDense(len(outputs), v.GetOutputSize(), nil)

	for i, output := range outputs {
		idx, ok := v.outputVocabulary[output]
		if ok {
			c.Set(i, idx, 1)
		}
	}

	return c
}

func (v *Vocabulary) GetInputMatrix(inputs []string) *mat.Dense {
	c := mat.NewDense(len(inputs), v.GetInputSize(), nil)

	for i, line := range inputs {
		for _, word := range strings.Split(strings.TrimSuffix(line, "\n"), " ") {
			word := strings.ToLower(word)
			idx, ok := v.inputVocabulary[word]

			if ok {
				c.Set(i, idx, 1)
			}
		}
	}

	return c
}

func (v *Vocabulary) GetOutput(idx int) string {
	for k, v := range v.outputVocabulary {
		if v+1 == idx {
			return k
		}
	}
	return ""
}

func getVocabulary(data []string) map[string]int {
	counter := map[string]int{}

	i := 0

	for _, line := range data {
		for _, word := range strings.Split(line, " ") {

			word := strings.ToLower(word)
			_, ok := counter[word]

			if !ok {
				counter[word] = i
				i += 1
			}
		}
	}

	return counter
}
