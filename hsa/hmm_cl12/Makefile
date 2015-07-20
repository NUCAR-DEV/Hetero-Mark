CC=g++
IFLAG=-I$(M2S_ROOT)/runtime/include -I$(HSA_LLVM_PATH)/../include -I$(HSA_RUNTIME_PATH)/include

all: HMM

HMM: hmm.cc kernels.o
	$(CC) hmm.cc kernels.o -L$(HSA_RUNTIME_PATH)/lib/ -lhsa-runtime64 $(IFLAG) -o HMM -std=c++11

kernels.o: kernels.cl
	snack.sh -c kernels.cl

clean:
	rm *.o HMM *.snackwrap.c *_brig.h isa.txt
