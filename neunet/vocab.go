package neunet

import (
	"encoding/gob"
	"log"
	"os"
	"strings"

	"gonum.org/v1/gonum/mat"
)

type vocab struct {
	data map[string]int
}

func NewVocab() *vocab {
	return &vocab{
		data: map[string]int{},
	}
}

func (v *vocab) ProcessText(txt string) {
	for _, word := range strings.Split(txt, " ") {
		v.AddWord(word)
	}
}

func (v *vocab) AddWord(word string) {
	_, ok := v.data[word]

	if !ok {
		v.data[word] = len(v.data)
	}
}

func (v *vocab) Size() int {
	return len(v.data)
}

func (v *vocab) OneHotEncode(txt string, m *mat.Dense, r int, threshold int) {
	for _, word := range strings.Split(txt, " ") {
		i, ok := v.data[word]

		if ok {
			m.Set(r, i, 1.)
		} else {
			if threshold > 0 {
				for k, i := range v.data {
					d := Levenshtein(&k, &word)
					if d < threshold {
						m.Set(r, i, 1)
						break
					}
				}
			}
		}
	}
}

func (v *vocab) IndexEncode(txt string, m *mat.Dense, r int) {
	i, ok := v.data[txt]

	if ok {
		m.Set(0, r, float64(i))
	}
}

func (v *vocab) OutputDecode(data *mat.Dense) (string, float64) {
	idx, prob := Argmax(data.RawMatrix().Data)

	for k, val := range v.data {
		if val == idx {
			return k, prob
		}
	}

	return "", 0.0
}

func (v *vocab) Save(path string) {
	f, err := os.Create(path)
	defer f.Close()

	if err == nil {
		e := gob.NewEncoder(f)
		e.Encode(v.data)
	}
}

func LoadVocab(path string) *vocab {
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

	return &vocab{
		data: data,
	}
}
