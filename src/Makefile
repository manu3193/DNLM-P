CUDA_HOME=/usr/local/cuda-10.1
OPENCV_HOME = /root/Manuel/opencv/

# define the compilers to use
PGI = pgcc
PGI++ = pgc++
# define any compile-time flags
CFLAGS_LINK_MC = -acc -g -Minfo=all -ta=multicore 
CFLAGS_MC = -acc -g -ta=multicore
CFLAGSPGI_MC = -acc -Minfo=all -g -ta=multicore #-Minline
LIBS_MC = -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc
LDFLAGS_MC = -L$(OPENCV_HOME)/lib


# define include path flags 
INCLUDES= -I../include -I$(OPENCV_HOME)/include 

# define the executable file
MAIN_MC = nlmfilter_multicore_fft

all: clean $(MAIN_MC)

.PHONY: clean

$(MAIN_MC): fft.o DNLMFilter_OpenACC.o ParallelDNLM.o  
	$(PGI++) $(CFLAGS_LINK_MC) -o $@ $^ $(LDFLAGS_MC) $(INCLUDES)  $(LIBS_MC)  

ParallelDNLM.o: ParallelDNLM_MC.cpp
	$(PGI++) $(CFLAGS_MC) $(INCLUDES) -c $< -o $@

DNLMFilter_OpenACC.o: DNLMFilter_OpenACC.c
	$(PGI++) $(CFLAGSPGI_MC) -c $< -o $@ $(INCLUDES)

fft.o: fft.c
	$(PGI++) $(CFLAGSPGI_MC) -c $< -o $@ $(INCLUDES)


clean:
	rm -rf $(MAIN) *.o 
