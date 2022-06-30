CXX = g++
NVCC = nvcc


GENCODE = -arch=compute_70 -code=sm_70


#CUDAFLAGS = $(GENCODE)  -g -lineinfo 
#CUDAFLAGS = $(GENCODE)  -g -G
CUDAFLAGS =  $(GENCODE)  -O3

CUDAFLAGS += -Xcompiler -fopenmp 
# cannot solve shared race problem, cannot compile with -G

# for Device Code
CUDA_PATH = /usr/local/cuda
#LDFLAGS += -L${CUDA_PATH}/lib64
#LDFLAGS += -lm -lcudart -lcusparse $(CUDAFLAGS)
LDFLAGS += -lcusparse $(CUDAFLAGS)
INCLUDE = -I./include -I./CUDATools -I./GPU 
INCLUDE += -I${CUDA_PATH}/include
INCLUDE += -I${CUDA_PATH}/samples/common/inc

BIN = ./bin
SRC = ./source
OBJ = ./obj
INC = ./include

OBJ_LIB = $(OBJ)/CSR.o $(OBJ)/Config.o $(OBJ)/COO.o $(OBJ)/DataLoader.o $(OBJ)/Executor.o $(OBJ)/RunConfig.o $(OBJ)/dCSR.o  $(OBJ)/cuSparseMultiply.o
OBJ_LIB2 = $(OBJ)/GPU/Compare.o $(OBJ)/GPU/Transpose.o $(OBJ)/GPU/memory.o $(OBJ)/GPU/Multiply.o 
#COMMON_DEP = cuda_common.h 

$(OBJ)/%.o : $(SRC)/%.cu $(INC)/%.h
	mkdir -p $(dir $@)
	$(NVCC) -c $(CUDAFLAGS) $(INCLUDE) -o $@ $<

$(OBJ)/%.o : $(SRC)/%.cu 
	mkdir -p $(dir $@)
	$(NVCC) -c $(CUDAFLAGS) $(INCLUDE) -o $@ $<

$(OBJ)/%.o : $(SRC)/%.cpp
	mkdir -p $(dir $@)
	$(NVCC) -c $(CUDAFLAGS) $(INCLUDE) -o $@ $<

$(OBJ)/%.o : $(SRC)/%.cpp $(INC)/%.h
	mkdir -p $(dir $@)
	$(NVCC) -c $(CUDAFLAGS) $(INCLUDE) -o $@ $<

speck : $(OBJ_LIB2) $(OBJ_LIB) $(OBJ)/runspECK.o
	$(NVCC)  -o $@ $^ $(LDFLAGS) $(INCLUDE)

reg_speck : $(OBJ_LIB2) $(OBJ_LIB) $(OBJ)/reg_runspECK.o
	$(NVCC)  -o $@ $^ $(LDFLAGS) $(INCLUDE)

all : speck reg_speck

clean :
	rm -rf $(BIN)/*
	rm -rf $(OBJ)/*
