package web

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"os"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
)

var ResponseSet map[string][]string

type Server struct {
	port int
}

func NewServer(port int) *Server {
	return &Server{
		port: port,
	}
}

func (s *Server) Run() {
	LoadResponseSet()

	r := chi.NewRouter()
	r.Use(middleware.Logger)
	r.Get("/", Index)
	r.Post("/api/chat", ChatHandler)

	fmt.Printf(fmt.Sprintf("Starting server at port %v\n", s.port))
	http.ListenAndServe(fmt.Sprintf(":%v", s.port), r)
}

func LoadResponseSet() {
	f, err := os.Open("data/responses.json")
	defer f.Close()

	if err != nil {
		log.Fatal(err)
	}

	byteValue, _ := ioutil.ReadAll(f)
	json.Unmarshal([]byte(byteValue), &ResponseSet)
}
