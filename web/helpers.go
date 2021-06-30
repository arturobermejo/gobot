package web

import (
	"encoding/json"
	"net/http"
)

func JsonResponse(w http.ResponseWriter, s interface{}, code int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(s)
}
