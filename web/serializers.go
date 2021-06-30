package web

type ChatRequest struct {
	Input string `json:"input"`
}

func (r *ChatRequest) Validate() map[string]string {
	errs := map[string]string{}

	if r.Input == "" {
		errs["input"] = "The field is required"
	}

	return errs
}

type ChatResponse struct {
	Input  string  `json:"input"`
	Intent string  `json:"intent"`
	Prob   float64 `json:"prob"`
	Output string  `json:"output"`
}
