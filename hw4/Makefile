CC = gcc
NVCC = nvcc

CFLAGS = -Wall -g
NVFLAGS = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61
LDFLAGS = -lm

TARGETS = hw4

all: $(TARGETS)

clean:
	rm -f $(TARGETS)

.PHONY: all clean

hw4: hw4.cu
	$(NVCC) $(NVFLAGS) -o $@ $? $(LDFLAGS)