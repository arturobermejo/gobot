package model

import (
	"bufio"
	"log"
	"os"
	"strings"

	"gonum.org/v1/gonum/mat"
)

type Vocab map[string]int

type VocabSet struct {
	in  Vocab
	out Vocab
}

type dataLoader struct {
	inputs   *mat.Dense
	outputs  *mat.Dense
	vocabset VocabSet
}

func NewDataLoader(path string) *dataLoader {
	inData, outData, inVocab, outVocab := GetData(path)

	return &dataLoader{
		inputs:  EncodeInputs(inData, inVocab, 0),
		outputs: EncodeOutputs(outData, outVocab),
		vocabset: VocabSet{
			in:  inVocab,
			out: outVocab,
		},
	}
}

func GetData(path string) ([]string, []string, Vocab, Vocab) {
	f, err := os.Open(path)

	if err != nil {
		log.Fatal(err)
	}

	defer f.Close()

	s := bufio.NewScanner(f)

	s.Split(bufio.ScanLines)

	inData := []string{}
	outData := []string{}
	inVocab := map[string]int{}
	outVocab := map[string]int{}

	for s.Scan() {
		line := s.Text()
		sample := strings.Split(line, "|")
		input := CleanText(sample[0])
		output := sample[1]

		for _, word := range strings.Split(input, " ") {
			_, ok := inVocab[word]
			if !ok {
				inVocab[word] = len(inVocab)
			}
		}

		_, ok := outVocab[output]
		if !ok {
			outVocab[output] = len(outVocab)
		}

		inData = append(inData, input)
		outData = append(outData, output)
	}

	return inData, outData, inVocab, outVocab
}

func EncodeInputs(data []string, vc Vocab, threshold int) *mat.Dense {
	o := mat.NewDense(len(data), len(vc), nil)

	for i, input := range data {
		words := strings.Split(CleanText(input), " ")

		for _, word := range words {
			idx, ok := vc[word]

			if ok {
				o.Set(i, idx, 1)
			} else {
				for k, v := range vc {
					d := Levenshtein(&k, &word)
					if d < threshold {
						o.Set(i, v, 1)
						break
					}
				}
			}
		}
	}

	return o
}

func EncodeOutputs(data []string, vc Vocab) *mat.Dense {
	o := mat.NewDense(1, len(data), nil)

	for i, output := range data {
		o.Set(0, i, float64(vc[output]))
	}

	return o
}
