package neunet

import (
	"log"
	"regexp"
	"strings"
	"unicode"

	"golang.org/x/text/runes"
	"golang.org/x/text/transform"
	"golang.org/x/text/unicode/norm"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/gonum/stat/distuv"
)

// Return a random slice with samples from a normal distribution
func RandomSlice(size int) []float64 {
	dist := distuv.Normal{
		Mu:    0.0,
		Sigma: 0.05,
	}

	r := make([]float64, size)
	for i := 0; i < size; i++ {
		r[i] = dist.Rand()
	}
	return r
}

// Get mean accuracy between two matrix
func Accuracy(m, n *mat.Dense) float64 {
	r, _ := m.Dims()
	result := mat.NewDense(1, r, nil)

	for i := 0; i < r; i++ {
		mi, _ := Argmax(m.RawRowView(i))
		ni, _ := Argmax(n.RawRowView(i))

		if mi == ni {
			result.Set(0, i, 1)
		}
	}

	return stat.Mean(result.RawMatrix().Data, nil)
}

// Get the max value of an array and the corresponding index
func Argmax(s []float64) (int, float64) {
	var idx int

	n := len(s)
	max := s[0]
	idx = 0

	for i := 1; i < n; i++ {
		v := s[i]
		if v > max {
			max = v
			idx = i
		}
	}

	return idx, max
}

func RemoveNoLetters(s string) string {
	reg, err := regexp.Compile("[^a-zA-Z ]+")
	if err != nil {
		log.Fatal(err)
	}
	return reg.ReplaceAllString(s, "")
}

func RemoveAccents(s string) string {
	t := transform.Chain(norm.NFD, runes.Remove(runes.In(unicode.Mn)), norm.NFC)
	output, _, err := transform.String(t, s)
	if err != nil {
		log.Fatal(err)
	}
	return output
}

func CleanText(s string) string {
	return strings.ToLower(RemoveNoLetters(RemoveAccents(s)))
}
