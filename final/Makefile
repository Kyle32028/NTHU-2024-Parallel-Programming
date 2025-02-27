CC = gcc
CXX = g++
NVCC = nvcc
CXXFLAGS = -O3
CFLAGS = -O3
NVCCFLAGS = -O3 -Xcompiler -fopenmp
LDFLAGS = -lpng -lz
CUDA_LDFLAGS = -lpng -lcuda -lcudart

TARGETS = seq fft

.PHONY: all
all: $(TARGETS)

# 目標 1: 編譯 seq.cpp
seq: seq.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

# 目標 2: 編譯 fft.cu
fft: fft.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(CUDA_LDFLAGS)

# 為了方便使用，加入個別的編譯目標
.PHONY: cuda
cuda: fft

.PHONY: clean
clean:
	rm -f $(TARGETS)