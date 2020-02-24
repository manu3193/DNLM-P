CUDA_HOME=/opt/tools/cuda-10.1.105
OPENCV_HOME = /opt/tools/opencv-3

# define the compilers to use
CC = nvcc 
PGI = pgcc
PGI++ = pgc++

# define any compile-time flags
CFLAGS_DEBUG = -g -G -std=c++11 -cudart static -O0 
CFLAGS = -std=c++11 -ccbin=$(PGI++) -Xcompiler "-ta=tesla:cc35 -ta=tesla:nordc -Mcuda" -cudart static -use_fast_math -O3
CFLAGSPGI = -acc -Minfo=accel -ta=tesla:cc35 -ta=tesla:nordc -fast -Minline -Mvect -O3
CFLAGSPGI_DEBUG = -acc -g -Minfo=accel  -ta=tesla:nordc -ta=tesla:cc35 -Mcuda -Minline -Mvect -O0
LIBS = -lnppial_static -lnppitc_static -lnppidei_static -lnppif_static -lnppisu_static -lnppist_static -lnpps_static -lnppc_static -lculibos -lopencv_core -lopencv_highgui -lopencv_imgcodecs
LDFLAGS = -L$(CUDA_HOME)/lib64 -L$(OPENCV_HOME)/lib

# define include path flags 
INCLUDES= -I../include -I$(CUDA_HOME)/include -I$(OPENCV_HOME)/include 

# define the executable file
MAIN = nlmfilter_k40_fft

all: clean $(MAIN)

.PHONY: clean

$(MAIN):  DNLMFilter_OpenACC.o ParallelDNLM.o 
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)  

ParallelDNLM.o: ParallelDNLM.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

DNLMFilter_OpenACC.o: DNLMFilter_OpenACC.c
	$(PGI) $(CFLAGSPGI) -c $< -o $@

clean:
	rm -rf $(MAIN) *.o 
