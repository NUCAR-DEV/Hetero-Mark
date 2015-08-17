CC=g++
CFLAG=-std=c++0x -g
IFLAG=-I$(HSA_LLVM_PATH)/../include -I$(HSA_RUNTIME_PATH)/include

all: KMEANS

KMEANS: kmeans.cc kmeans.h kernels.o
	$(CC) kmeans.cc kernels.o \
		$(IFLAG) $(CFLAG) \
		-L$(HSA_RUNTIME_PATH)/lib/ -lhsa-runtime64 \
 		-o KMEANS

kernels.o: kernels.cl
	snack.sh -c kernels.cl

clean:
	rm *.o KMEANS *.snackwrap.c *_brig.h isa.txt
