CC=g++
CFLAG=-std=c++0x -g
IFLAG=-I$(HSA_LLVM_PATH)/../include -I$(HSA_RUNTIME_PATH)/include

BENCHMARK=pagerank

all: $(BENCHMARK)

$(BENCHMARK): pagerank.cc pagerank.h kernels.o
	$(CC) pagerank.cc kernels.o \
		$(IFLAG) $(CFLAG) \
		-L$(HSA_RUNTIME_PATH)/lib/ -lhsa-runtime64 \
 		-o $@

kernels.o: kernels.cl
	snack.sh -c kernels.cl

clean:
	rm *.o $(BENCHMARK) *.snackwrap.c *_brig.h isa.txt
