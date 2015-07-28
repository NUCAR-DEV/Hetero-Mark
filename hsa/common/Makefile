CPPCHECK=cppcheck
HAVE_CPPCHECK=$(if $(shell which $(CPPCHECK)),TRUE)

CPPLINT=cpplint.py
HAVE_CPPLINT=$(if $(shell which $(CPPLINT)),TRUE)

CXX=g++
CXXFLAG=-std=c++0x

HEADERS=Benchmark.h \
	OptionParser.h \
	TimeKeeper.h

OBJS=OptionParser.o

all: common.a

%.o: %.cc $(HEADERS)
	echo here
ifeq ($(HAVE_CPPCHECK), TRUE)
	$(CPPCHECK) $^
endif
ifeq ($(HAVE_CPPLINT), TRUE)
	$(CPPLINT) $^
endif
	$(CXX) $(CXXFLAG) -c -o $@ $<

common.a: $(OBJS)
	ar rcs $@ $^
	

clean:
	rm *.o *.a


	
