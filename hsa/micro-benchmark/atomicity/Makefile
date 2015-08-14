CPPCHECK=cppcheck
HAVE_CPPCHECK=$(if $(shell which $(CPPCHECK)),TRUE)

CPPLINT=cpplint.py
HAVE_CPPLINT=$(if $(shell which $(CPPLINT)),TRUE)

SNACK=snack.sh
HAVE_SNACK=$(if $(shell which $(SNACK)),TRUE)

IFLAG=-I../../../ -I../../gtest/include

GTEST=../../gtest/gtest_main.a

CXX=g++
CXXFLAG=-std=c++0x -Wall -Werror -pthread -O0 -g
HSA_RUNTIME=-L$(HSA_RUNTIME_PATH)/lib -lhsa-runtime64

HEADERS=AtomicityBenchmark.h

SRCS=main.cc \
     AtomicityBenchmark.cc

UNIT_TESTS=

OBJS=main.o \
     AtomicityBenchmark.o

all: run

.PHONY: clean cpp_check cpp_lint check unittest

check: cpp_lint cpp_check all unittest

# Define cppcheck rules
cpp_check: $(HEADERS) $(SRCS)
ifeq ($(HAVE_CPPCHECK), TRUE)
	$(CPPCHECK) $(HEADERS) $(SRCS)
else
	$(warning "cppcheck is not found")
endif

#define cpplint rules
cpp_lint: $(HEADERS) $(SRCS)
ifeq ($(HAVE_CPPLINT), TRUE)
	$(CPPLINT) $(HEADERS) $(SRCS)
else
	$(warning "cpplint is not found")
endif

unittest: 

kernels.o: kernels.cl
ifeq ($(HAVE_SNACK), TRUE)
	$(SNACK) -c -gccopt 0 -opt 0 $^
else
	$(error snack is not properly installed)
endif

%.o: %.cc $(HEADERS)
	$(CXX) $(CXXFLAG) $(IFLAG) -c -o $@ $<

run: kernels.o $(OBJS)
	$(CXX) $(CXXFLAG) $^ ../../common/common.a $(HSA_RUNTIME) -o $@

clean:
	rm *.o *.a kernels.h
