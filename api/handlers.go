package api

import (
	"encoding/json"
	"net/http"
)

type UserMessage struct {
	Input string `json:"input"`
}

type UserResponse struct {
	Input  string `json:"input"`
	Intent string `json:"intent"`
	Output string `json:"output"`
}

func ChatHandler(w http.ResponseWriter, r *http.Request) {
	var m UserMessage

	err := json.NewDecoder(r.Body).Decode(&m)

	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	intent, output := ChatService(m.Input)

	res := UserResponse{m.Input, intent, output}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)

	json.NewEncoder(w).Encode(res)
}
