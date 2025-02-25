CXX   = g++
NVCC  = nvcc

CXXFLAGS = -O3 -fopenmp -funroll-loops -march=native
NVFLAGS  = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 -Xcompiler -fopenmp
LDLIBS   = -lm

TARGETS = hw3-1 hw3-2 hw3-3

.PHONY: all clean

all: $(TARGETS)

hw3-1: hw3-1.cc
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

hw3-2: hw3-2.cu
	$(NVCC) $(NVFLAGS) -o $@ $^ $(LDLIBS)

hw3-3: hw3-3.cu
	$(NVCC) $(NVFLAGS) -o $@ $^ $(LDLIBS)

clean:
	rm -f $(TARGETS)
