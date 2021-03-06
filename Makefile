.SILENT:

# Source files
CUDA_FILES = $(wildcard src/gpu/*.cu) 
gDel2D_CUDA_FILES = $(wildcard lib/gDel2D/src/*.cu)
COMMON_CPP_FILES = $(wildcard src/common/*.cpp)
GPU_CPP_FILES := $(wildcard src/gpu/*.cpp) $(COMMON_CPP_FILES)
gDel2D_CPP_FILES = $(wildcard lib/gDel2D/src/*.cpp)
CPU_CPP_FILES := $(wildcard src/cpu/*.cpp) $(COMMON_CPP_FILES)


CUDA_OBJ = cuda.o

# CUDA Compiler and Flags
CUDA_PATH = /usr/local/cuda
CUDA_INC_PATH = $(CUDA_PATH)/include
CUDA_BIN_PATH = $(CUDA_PATH)/bin
CUDA_LIB_PATH = $(CUDA_PATH)/lib64

NVCC = $(CUDA_BIN_PATH)/nvcc

# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
NVCC_FLAGS := -m32
else
NVCC_FLAGS := -m64
endif
NVCC_FLAGS += -g -dc -Wno-deprecated-gpu-targets --std=c++11 \
             --expt-relaxed-constexpr

NVCC_GENCODES = -gencode arch=compute_30,code=sm_30 \
		-gencode arch=compute_35,code=sm_35 \
		-gencode arch=compute_50,code=sm_50 \
		-gencode arch=compute_52,code=sm_52 \
		-gencode arch=compute_60,code=sm_60 \
		-gencode arch=compute_61,code=sm_61 \
		-gencode arch=compute_61,code=compute_61

# CUDA Object Files
CUDA_OBJ_FILES = $(notdir $(addsuffix .o, $(CUDA_FILES)))
gDel2D_CUDA_OBJ_FILES = $(notdir $(addsuffix .o, $(gDel2D_CUDA_FILES)))
# ------------------------------------------------------------------------------

# CUDA Linker and Flags
CUDA_LINK_FLAGS = -dlink -Wno-deprecated-gpu-targets

# ------------------------------------------------------------------------------

# C++ Compiler and Flags
GPP = g++
FLAGS = -g -Wall -D_REENTRANT -std=c++0x -pthread -O3 -DNDEBUG
GPU_INCLUDE = -I$(CUDA_INC_PATH) -I./src/common -I./lib/cudpp/include -I./lib/gDel2D/src 
LIBS = -L$(CUDA_LIB_PATH) -lcudart -lcuda -L./lib/cudpp/build/lib -lcudpp_hash -lcudpp
CPU_INCLUDE = -I./include -I./src/common

NVCC_INCLUDE = $(GPU_INCLUDE)

# C++ Object Files
OBJ_CPU = $(addprefix cpu-, $(notdir $(addsuffix .o, $(CPU_CPP_FILES))))
OBJ_GPU = $(addprefix gpu-, $(notdir $(addsuffix .o, $(GPU_CPP_FILES))))
OBJ_gDel2D = $(addprefix gdel-, $(notdir $(addsuffix .o, $(gDel2D_CPP_FILES))))

all: cpu gpu

cpu: $(OBJ_CPU)
	$(GPP) $(FLAGS) -o cpu-dbscan $(CPU_INCLUDE) $^

gpu: $(OBJ_GPU) $(CUDA_OBJ) $(CUDA_OBJ_FILES) $(OBJ_gDel2D) $(gDel2D_CUDA_OBJ_FILES)
	$(NVCC) -g -D_REENTRANT -Xcompiler -pthread -O3 -DNDEBUG -o gpu-dbscan $(GPU_INCLUDE) $^ $(LIBS) 


# Compile C++ Source Files
cpu-%.cpp.o: src/cpu/%.cpp
	$(GPP) $(FLAGS) -c -o $@ $(CPU_INCLUDE) $< 

cpu-%.cpp.o: src/common/%.cpp
	$(GPP) $(FLAGS) -c -o $@ $(CPU_INCLUDE) $<

gpu-%.cpp.o: src/gpu/%.cpp
	$(GPP) $(FLAGS) -c -o $@ $(GPU_INCLUDE) $< 

gpu-%.cpp.o: src/common/%.cpp
	$(GPP) $(FLAGS) -c -o $@ $(GPU_INCLUDE) $<

gdel-%.cpp.o: lib/gDel2D/src/%.cpp
	$(GPP) $(FLAGS) -c -o $@ $(GPU_INCLUDE) $<

# Compile CUDA Source Files
%.cu.o: src/gpu/%.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_GENCODES) -c -o $@ $(NVCC_INCLUDE) $<

%.cu.o: lib/gDel2D/src/%.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_GENCODES) -c -o $@ $(NVCC_INCLUDE) $<

# cuda: $(CUDA_OBJ_FILES) $(CUDA_OBJ) $(gDel2D_CUDA_OBJ_FILES)

# Make linked device code
$(CUDA_OBJ): $(CUDA_OBJ_FILES) $(gDel2D_CUDA_OBJ_FILES) $(OBJ_GPU) $(OBJ_gDel2D)
	$(NVCC) $(CUDA_LINK_FLAGS) $(NVCC_GENCODES) -o $@ $(NVCC_INCLUDE) $^


cudpp:
	rm -rf lib/cudpp/build
	mkdir lib/cudpp/build
	cd lib/cudpp/build; cmake ../src ../; make

correctness: cpu gpu
	./gpu-dbscan -p > gpu_res.txt
	./cpu-dbscan -p > cpu_res.txt
	python3 scripts/correctness.py cpu_res.txt gpu_res.txt
	echo 'If there are no errors above, GPU and CPU DBSCAN returned the same result'
	rm -f gpu_res.txt cpu_res.txt


time-trial: cpu gpu
	echo 'GPU:'
	for pts in 10000 15000 20000; do echo "n=$$pts"; ./gpu-dbscan -n $$pts; done; 
	echo 'CPU:'
	for pts in 10000 15000 20000; do echo "n=$$pts"; ./cpu-dbscan -n $$pts; done; 

# Clean everything including temporary Emacs files
clean:
	rm -f cpu-dbscan gpu-dbscan *.o *~
	rm -f src/*~ lib/gDel2D/src/*~

.PHONY: clean cudpp 
