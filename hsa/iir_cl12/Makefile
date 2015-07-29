CC=g++
CFLAG=-std=c++0x -g
IFLAG=-I$(HSA_LLVM_PATH)/../include -I$(HSA_RUNTIME_PATH)/include

all: IIR

IIR: iir.cc kernels.o
	$(CC) iir.cc kernels.o -L$(HSA_RUNTIME_PATH)/lib/ -lhsa-runtime64 $(IFLAG) -o IIR $(CFLAG)

kernels.o: kernels.cl
	snack.sh -c kernels.cl

clean:
	rm *.o IIR *.snackwrap.c *_brig.h isa.txt
