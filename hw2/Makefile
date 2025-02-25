# Compiler definitions for hw2a
HW2A_CXX = g++

# Compiler definitions for hw2b
HW2B_CXX = mpicxx

# Common libraries and flags
LDLIBS = -lpng
BASE_FLAGS = -lm -O1 -pthread -mavx512f -mavx512dq -mfma

# Specific flags
HW2A_FLAGS = $(BASE_FLAGS)
HW2B_FLAGS = $(BASE_FLAGS) -fopenmp

# All targets
TARGETS = hw2a hw2b

.PHONY: all clean

all: $(TARGETS)

# Rules for hw2a
hw2a: hw2a.o
	$(HW2A_CXX) $(HW2A_FLAGS) -o $@ $^ $(LDLIBS)

hw2a.o: hw2a.cc
	$(HW2A_CXX) $(HW2A_FLAGS) -c $<

# Rules for hw2b
hw2b: hw2b.o
	$(HW2B_CXX) $(HW2B_FLAGS) -o $@ $^ $(LDLIBS)

hw2b.o: hw2b.cc
	$(HW2B_CXX) $(HW2B_FLAGS) -c $<

clean:
	rm -f $(TARGETS) $(TARGETS:=.o)