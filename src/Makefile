#CUDA_HOME=/opt/tools/cuda-10.1.105/
CUDA_HOME = /opt/tools/cuda-9.0.176
#CUDA_HOME = /opt/tools/cuda-8.0
# define the C compiler to use
CC = nvcc 
PGI = pgc++
# define any compile-time flags
CFLAGS_DEBUG = -g -G -ccbin=pgc++ -Xcompiler "-acc -g -std=c++11 -Minfo=accel  -ta=tesla:nordc -ta=tesla:cc35 -ta=tesla -Mcuda -Minline -Mvect " -cudart static -O0 -DDEBUG
CFLAGS = -ccbin=pgc++ -Xcompiler  "-acc -std=c++11 -Minfo=accel  -ta=tesla:cc35 -ta=tesla:nordc -fast -Mcuda -Minline -Mvect -O3"  -cudart static -use_fast_math -O3
CFALGS_MC = -g -std=c++11 -Minfo=all -acc -ta=multicore -O0
LIBS = -lnppial_static -lnppitc_static -lnppidei_static -lnppif_static -lnppisu_static -lnppist_static -lnpps_static -lnppc_static -lculibos -lopencv_core -lopencv_highgui -lopencv_imgcodecs
LIBS_MC = -lopencv_core -lopencv_highgui -lopencv_imgcodecs
#LIBS = -lnppc_static -lnppi_static -lnpps_static -lculibos -lopencv_core -lopencv_highgui -lopencv_imgcodecs
LDFLAGS = -L$(CUDA_HOME)/lib64 -L/opt/tools/opencv-3/lib
LDFLAGS_DEBUG = -L$(CUDA_HOME)/lib64 -L/opt/tools/opencv-3/lib
LDFLAGS_MC = -L/opt/tools/opencv-3/lib
# define any directories containing header files other than /usr/include
#
INCLUDES= -I../include -I$(CUDA_HOME)/include -I/opt/tools/opencv-3/include 
INCLUDES_MC = -I/opt/tools/opencv-3/include
# define the C source files
SRCS =  ParallelDNLM.cpp  DNLMFilter_OpenACC.cpp 
SRCS_MC =  ParallelDNLM_MC.cpp  DNLMFilter_OpenACC.cpp
# define the C object files
#
# This uses Suffix Replacement within a macro:
#   $(name:string1=string2)
#         For each word in 'name' replace 'string1' with 'string2'
# Below we are replacing the suffix .c of all words in the macro SRCS
# with the .o suffix
#
OBJS = $(SRCS:.c=.o)
OBJS_MC = $(SRCS_MC:.c=.o)
# define the executable file
MAIN = dnlmfilter
#
# The following part of the makefile is generic; it can be used to
# build any executable just by changing the definitions above and by
# deleting dependencies appended to the file from 'make depend'
#
.PHONY: clean
$(MAIN): $(OBJS) 
	$(CC) $(OBJS) $(CFLAGS) $(INCLUDES)  -o $@ $(LDFLAGS) $(LIBS)
multicore: $(OBJS_MC)  
	$(PGI) $(OBJS_MC) $(CFLAGS_MC) $(INCLUDES_MC)  -o $@ $(LDFLAGS_MC) $(LIBS_MC)
debug: $(OBJS) 
	$(CC) $(OBJS) $(CFLAGS_DEBUG) $(INCLUDES)  -o $(MAIN) $(LDFLAGS_DEBUG) $(LIBS)
#this is a suffix replacement rule for building .o's from .c's
# it uses automatic variables $<: the name of the prerequisite of
# the rule(a .c file) and $@: the name of the target of the rule (a .o file)
# (see the gnu make manual section about automatic variables)
.c.o:
	pgc++ -c $(PGFLAGS) $(INCLUDES) $<
clean:
	rm $(MAIN) $(OBJS)
