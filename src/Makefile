OS=$(shell uname -s)
ifeq ($(OS),Darwin)
	PLATFORM?=mac
else
	PLATFORM?=gcc
endif

include Makefile.in.$(PLATFORM)

OPTFLAGS = -ffast-math -fassociative-math -freciprocal-math -fno-signed-zeros \
-Ofast -frename-registers -funroll-loops -fmodulo-sched \
-fmodulo-sched-allow-regmoves -ftree-vectorize -march=native \
-flax-vector-conversions

CFLAGS = -std=c++11 -fopenmp
LDFLAGS = -fopenmp

SRC = $(shell find . -name "*.cpp")
OBJ = $(SRC:.cpp=.o)

main: $(OBJ)
	$(LD) $(LDFLAGS) $(LIBBLAS) -o $@ $^

%.o: %.cpp
	$(CC) $(CFLAGS) $(OPTFLAGS) $(INCBLAS) -Wall -MMD -c -o $@ $<

clean:
	rm -f main *.o *.d

# Optimize build with auto-generated dependencies.
-include $(SRC:%.cpp=%.d)
