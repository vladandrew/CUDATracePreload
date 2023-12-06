LIBTORCH_CUDA_SO ?= $(shell find ${HOME} -name 'libtorch_cuda.so' | head -n 1)
CUDA_INCLUDE_PATH ?= /usr/local/cuda/include
NCCL_INCLUDE_PATH ?= /usr/local/nccl/include
TRACK_CUDA ?= 1
TRACK_NCCL ?= 0

CXX ?= g++
CXXFLAGS := -fPIC -shared -DTRACK_CUDA=$(TRACK_CUDA) -DTRACK_NCCL=$(TRACK_NCCL) -DLIBTORCH_CUDA_PATH=\"$(LIBTORCH_CUDA_SO)\"
INCLUDES := -I$(CUDA_INCLUDE_PATH) -I$(NCCL_INCLUDE_PATH)

build: tracer.cpp helpers.h
	@if [ -z "$(LIBTORCH_CUDA_SO)" ]; then \
		echo "libtorch_cuda.so not found. Please specify the path through LIBTORCH_CUDA_SO (make LIBTORCH_CUDA_SO=/path)"; \
		exit 1; \
	fi
	$(CXX) $(INCLUDES) $(CXXFLAGS) -o tracer.so tracer.cpp -ldl
