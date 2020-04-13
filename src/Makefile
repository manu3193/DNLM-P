CUDA_HOME=/opt/tools/cuda-10.1.105
OPENCV_HOME = /opt/tools/opencv-3

# define the compilers to use
CC = nvcc 
PGI = pgcc
PGI++ = pgc++
# define any compile-time flags
CFLAGS_LINK = -acc -g -Minfo=all -ta=tesla:cc35 -ta=tesla:lineinfo -Mcuda -O0 
CFLAGS = -std=c++11 -g -G -arch=sm_35 -rdc=true -ccbin=$(PGI++) -Xcompiler "-ta=tesla:cc35 -ta=tesla:rdc -Mcuda" -cudart static -O0# -use_fast_math -O3
CFLAGSPGI = -acc -Minfo=all -g -ta=tesla:cc35 -ta=tesla:lineinfo -Minline -O0 #-Mvect -fast -O3
LIBS = -lnppial_static -lnppitc_static -lnppidei_static -lnppif_static -lnppisu_static -lnppist_static -lnpps_static -lnppc_static -lculibos -lopencv_core -lopencv_highgui -lopencv_imgcodecs
LDFLAGS = -L$(CUDA_HOME)/lib64 -L$(OPENCV_HOME)/lib

# define include path flags 
INCLUDES= -I../include -I$(CUDA_HOME)/include -I$(OPENCV_HOME)/include 

# define the executable file
MAIN = nlmfilter_k40_fft

all: clean $(MAIN)

.PHONY: clean

$(MAIN): fft.o DNLMFilter_OpenACC.o ParallelDNLM.o  
	$(PGI++) $(CFLAGS_LINK) -o $@ $^ $(LDFLAGS) $(INCLUDES)  $(LIBS)  

ParallelDNLM.o: ParallelDNLM.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -dc $< -o $@

DNLMFilter_OpenACC.o: DNLMFilter_OpenACC.c
	$(PGI) $(CFLAGSPGI) -c $< -o $@ $(INCLUDES)

fft.o: fft.c
	$(PGI) $(CFLAGSPGI) -c $< -o $@ $(INCLUDES)

clean:
	rm -rf $(MAIN) *.o 
