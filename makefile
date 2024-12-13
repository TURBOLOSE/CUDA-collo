.PHONY: clean, time
all: main 

main: main.cu rhs/VDP.cuh kernel.cuh collo.cuh
	nvc++ -std=c++23 -pg -O3 main.cu  -o main 
clean: 
	rm -f main
time: main
	time ./main
