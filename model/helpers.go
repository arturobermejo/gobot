package model

import (
	"log"
	"regexp"
	"strings"
	"unicode"

	"golang.org/x/text/runes"
	"golang.org/x/text/transform"
	"golang.org/x/text/unicode/norm"
)

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
