package web

import (
	"encoding/json"
	"net/http"
	"text/template"
)

type UserMessage struct {
	Input string `json:"input"`
}

type UserResponse struct {
	Input  string  `json:"input"`
	Intent string  `json:"intent"`
	Prob   float64 `json:"prob"`
	Output string  `json:"output"`
}

func ChatHandler(w http.ResponseWriter, r *http.Request) {
	var m UserMessage

	err := json.NewDecoder(r.Body).Decode(&m)

	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	intent, prob, output := ChatService(m.Input)

	res := UserResponse{m.Input, intent, prob, output}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)

	json.NewEncoder(w).Encode(res)
}

func Index(w http.ResponseWriter, r *http.Request) {
	t, err := template.New("index.html").Delims("[[", "]]").ParseFiles("static/index.html")

	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}

	t.Delims("[[", "]]")
	t.Execute(w, nil)
}
