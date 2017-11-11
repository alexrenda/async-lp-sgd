OS=$(shell uname -s)
ifeq ($(OS),Darwin)
	PLATFORM?=mac
else
	PLATFORM?=gcc
endif

include Makefile.in.$(PLATFORM)

# Possible flags to use later
# -ffast-math -Ofast -fassociative-math -freciprocal-math \
# -fno-signed-zeros -frename-registers -funroll-loops -fmodulo-sched \
# -fmodulo-sched-allow-regmoves -ftree-vectorizen

OPTFLAGS = -O3 -march=native

CFLAGS = -std=c++14

SRC = $(shell find . -name "*.cpp")
OBJ = $(SRC:.cpp=.o)

main: $(OBJ)
	$(LD) $(LIBBLAS) -o $@ $^

%.o: %.cpp
	$(CC) $(CFLAGS) $(OPTFLAGS) $(INCBLAS) -Wall -MMD -c -o $@ $<

clean:
	rm -f main *.o *.d

# Optimize build with auto-generated dependencies.
-include $(SRC:%.cpp=%.d)
