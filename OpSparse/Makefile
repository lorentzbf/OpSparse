CXX = g++
NVCC = nvcc

#CUDAFLAGS = $(GENCODE)  -g -lineinfo 
CUDAFLAGS =  $(GENCODE)  -O3

#CUDAFLAGS = $(GENCODE)  -g -G


GENCODE = -arch=compute_70 -code=sm_70
CUDAFLAGS += -Xcompiler -fopenmp 

# for Device Code
CUDA_PATH = /usr/local/cuda
LDFLAGS += -lcusparse $(CUDAFLAGS)
INCLUDE += -I${CUDA_PATH}/include
INCLUDE += -I${CUDA_PATH}/samples/common/inc
INCLUDE += -I./inc

BIN = ./bin
SRC = ./src
OBJ = ./obj
INC = ./inc

OBJ_LIB = $(OBJ)/Meta.o $(OBJ)/CSR.o $(OBJ)/Timings.o

COMMON_DEP = $(INC)/cuda_common.h $(INC)/define.h 

$(OBJ)/%.o : $(SRC)/%.cu $(INC)/%.h $(COMMON_DEP)
	mkdir -p $(dir $@)
	@echo $^
	$(NVCC) -c $(CUDAFLAGS) $(INCLUDE) -o $@ $<

$(OBJ)/%.o : $(SRC)/%.cu $(COMMON_DEP) 
	mkdir -p $(dir $@)
	@echo $^
	$(NVCC) -c $(CUDAFLAGS) $(INCLUDE) -o $@ $<

opsparse : $(OBJ_LIB) $(OBJ)/opsparse.o
	$(NVCC)  -o $@ $^ $(LDFLAGS) $(INCLUDE)

reg_opsparse : $(OBJ_LIB) $(OBJ)/reg_opsparse.o
	$(NVCC)  -o $@ $^ $(LDFLAGS) $(INCLUDE)

reg_cusparse : $(OBJ_LIB) $(OBJ)/reg_cusparse.o
	$(NVCC)  -o $@ $^ $(LDFLAGS) $(INCLUDE)

clean :
	rm -rf $(BIN)
	rm -rf $(OBJ)
