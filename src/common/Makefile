CPPCHECK=cppcheck
HAVE_CPPCHECK=$(if $(shell which $(CPPCHECK)),TRUE)

CPPLINT=cpplint.py
HAVE_CPPLINT=$(if $(shell which $(CPPLINT)),TRUE)

IFLAG=-I../../ -I../gtest/include

GTEST=../gtest/gtest_main.a

CXX=g++
CXXFLAG=-std=c++0x -Wall -Werror -pthread -g

HEADERS=Argument.h \
	ArgumentValue.h\
	ArgumentValueFactory.h \
	Benchmark.h \
	BenchmarkRunner.h \
	CommandLineOption.h \
	OptionParser.h \
	OptionParserImpl.h \
	OptionSetting.h \
	OptionSettingHelpPrinter.h \
	OptionSettingImpl.h \
	TimeMeasurement.h \
	TimeKeeper.h \
	TimeKeeperImpl.h \
	TimeKeeperSummaryPrinter.h \
	Timer.h \
	TimerImpl.h

SRCS=BenchmarkRunner.cc \
     CommandLineOption.cc \
     OptionParserImpl.cc \
     OptionSettingImpl.cc \
     OptionSettingHelpPrinter.cc \
     TimeMeasurement.cc \
     TimeKeeperImpl.cc \
     TimeKeeperSummaryPrinter.cc \
     TimerImpl.cc

UNIT_TESTS=BenchmarkRunner_unittest.o \
           OptionParserImpl_unittest.o \
           OptionSettingImpl_unittest.o \
	   OptionSettingHelpPrinter_unittest.o \
	   TimeKeeperImpl_unittest.o

OBJS=BenchmarkRunner.o \
     CommandLineOption.o \
     OptionParserImpl.o \
     OptionSettingImpl.o \
     OptionSettingHelpPrinter.o \
     TimeMeasurement.o \
     TimeKeeperImpl.o \
     TimeKeeperSummaryPrinter.o \
     TimerImpl.o

all: common.a

.PHONY: clean cpp_check cpp_lint check unittest

check: cpp_lint cpp_check all unittest

# Define cppcheck rules
ifeq ($(HAVE_CPPCHECK), TRUE)
cpp_check:  $(HEADERS) $(SRCS)
	$(CPPCHECK) $(HEADERS) $(SRCS)
else
cpp_check:
	echo "Warning: cppcheck is not available."
endif

#define cpplint rules
ifeq ($(HAVE_CPPLINT), TRUE)
cpp_lint: $(HEADERS) $(SRCS)
	$(CPPLINT) $(HEADERS) $(SRCS)
else
cpp_lint:
	echo "Warning: cpplint is not available"
endif

unittest: common.a $(UNIT_TESTS) $(OBJS)
	g++ $^ $(GTEST) -o $@ $(CXXFLAG) $(IFLAG)
	./$@


%.o: %.cc $(HEADERS)
	$(CXX) $(CXXFLAG) $(IFLAG) -c -o $@ $<

common.a: $(OBJS)
	ar rcs $@ $(OBJS)

clean:
	rm *.o *.a
