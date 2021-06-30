package web

import (
	"encoding/json"
	"net/http"
	"text/template"
)

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

func ChatHandler(w http.ResponseWriter, r *http.Request) {
	var req ChatRequest

	err := json.NewDecoder(r.Body).Decode(&req)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	errs := req.Validate()
	if len(errs) != 0 {
		JsonResponse(w, errs, http.StatusBadRequest)
		return
	}

	intent, prob, output := ChatService(req.Input)

	res := ChatResponse{req.Input, intent, prob, output}

	JsonResponse(w, res, http.StatusOK)
}

func Index(w http.ResponseWriter, r *http.Request) {
	t, err := template.New("index.html").Delims("[[", "]]").ParseFiles("static/index.html")

	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}

	t.Delims("[[", "]]")
	t.Execute(w, nil)
}
