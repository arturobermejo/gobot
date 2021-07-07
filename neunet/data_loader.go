package neunet

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type dataLoader struct {
	batch int
	cur   int
	end   bool
	x     *mat.Dense
	y     *mat.Dense
	n     int
}

func NewDataLoader(x, y *mat.Dense, batch int) *dataLoader {
	r, _ := x.Dims()
	n := int(math.Ceil(float64(r) / float64(batch)))

	return &dataLoader{
		x:     x,
		y:     y,
		n:     n,
		batch: batch,
	}
}

func (dl *dataLoader) Batch() bool {
	return !dl.end
}

func (dl *dataLoader) BatchCount() int {
	return dl.n
}

func (dl *dataLoader) Reset() {
	dl.cur = 0
	dl.end = false
}

func (dl *dataLoader) Data() (*mat.Dense, *mat.Dense) {
	if dl.end {
		return nil, nil
	}

	r, c := dl.x.Dims()

	e := dl.cur + dl.batch

	if e >= r {
		e = r
		dl.end = true
	}

	input := dl.x.Slice(dl.cur, e, 0, c).(*mat.Dense)
	output := dl.y.Slice(0, 1, dl.cur, e).(*mat.Dense)

	dl.cur = e

	return input, output
}
