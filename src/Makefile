ICC := $(shell command -v icc 2> /dev/null)
OS=$(shell uname -s)

ifdef ICC
	PLATFORM?=icc
	FLAGFILE?=icc_flags
else ifeq ($(OS),Darwin)
	PLATFORM?=mac
	FLAGFILE?=gcc_flags
else
	PLATFORM?=gcc
	FLAGFILE?=gcc_flags
endif

include Makefile.in.$(PLATFORM)
include Makefile.in.$(FLAGFILE)

SRC = $(shell find . -name "*.cpp")
OBJ = $(SRC:.cpp=.o)

main: $(OBJ)
	$(LD) $(LDFLAGS) $(LIBBLAS) -o $@ $^

%.o: %.cpp
	$(CC) $(CFLAGS) $(OPTFLAGS) $(INCBLAS) -Wall -MMD -c -o $@ $<

clean:
	rm -f main sgd svrg adam_serial adam_shared adam_private *.o *.d

# Optimize build with auto-generated dependencies.
-include $(SRC:%.cpp=%.d)
