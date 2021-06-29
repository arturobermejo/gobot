package main

import "github.com/arturobermejo/gobot/web"

func main() {
	s := web.NewServer(3000)
	s.Run()
}
