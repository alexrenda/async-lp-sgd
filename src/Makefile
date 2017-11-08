OS=$(shell uname -s)
ifeq ($(OS),Darwin)
	PLATFORM?=mac
else
	PLATFORM?=gcc
endif

include Makefile.in.$(PLATFORM)

OPTFLAGS = -Ofast -march=native -fassociative-math -freciprocal-math \
           -fno-signed-zeros -frename-registers -funroll-loops -fmodulo-sched \
           -fmodulo-sched-allow-regmoves -ftree-vectorize

CFLAGS = -std=gnu99
LDFLAGS = -lm

SRC = $(shell find . -name "*.c")
OBJ = $(SRC:.c=.o)

main: $(OBJ)
	$(LD) $(LDFLAGS) $(LIBBLAS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) $(INCBLAS) -Wall -MMD -c -o $@ $<

clean:
	rm -f main *.o *.d

# Optimize build with auto-generated dependencies.
-include $(SRC:%.c=%.d)
