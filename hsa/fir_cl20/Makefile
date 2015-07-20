CC=gcc
IFLAG=-I$(M2S_ROOT)/runtime/include -I$(HSA_LLVM_PATH)/../include -I$(HSA_RUNTIME_PATH)/include

all: FIR

FIR: fir.cc fir_Kernels.o
	$(CC) fir.cc fir_Kernels.o -L$(HSA_RUNTIME_PATH)/lib/ -lhsa-runtime64 $(IFLAG) -o FIR

fir_Kernels.o: fir_Kernels.cl
	snack.sh fir_Kernels.cl
	$(CC) $(IFLAG) fir_Kernels.snackwrap.c -c -o fir_Kernels.o

clean:
	rm *.o FIR *.snackwrap.c *_brig.h isa.txt
