.PHONY: clean, time
all: run_test 
	./run_test

run_test: test.cu rhs1.cuh kernel.cuh collo.cuh
	nvc++ -std=c++20 -g -pg -O3 test.cu  -o run_test 


consts: const.cpp temp.dat
	g++ -I. const.cpp  -std=c++20 -O3 -march=native -Wfatal-errors -o const
	./const

clean: 
	rm -f run_test
time: run_test
	time ./run_test
