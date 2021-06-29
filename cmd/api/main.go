package main

import "github.com/arturobermejo/gobot/api"

func main() {
	s := api.NewServer(3000)
	s.Run()
}
