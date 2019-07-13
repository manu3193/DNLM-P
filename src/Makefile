#
# 'make'        build executable file 'mycc'
# 'make clean'  removes all .o and executable files
#

#CUDA_HOME = /opt/tools/cuda-9.0.176
CUDA_HOME = /opt/tools/cuda-10.1.105
# define the C compiler to use
CC = nvcc 
# define any compile-time flags
CFLAGS_DEBUG = -g -O3 -cudart static -gencode arch=compute_35,code=sm_35 -use_fast_math
CFLAGS = -O3 -Xcompiler -fopenmp -cudart static -gencode arch=compute_35,code=sm_35 -use_fast_math

LIBS = -lnppc_static -lnppial_static -lnppitc_static -lnppidei_static -lnppif_static -lnppisu_static -lnpps_static -lculibos -lopencv_core -lopencv_highgui -lopencv_imgcodecs
#LIBS = -lnppc_static -lnppi_static -lnpps_static -lculibos -lopencv_core -lopencv_highgui -lopencv_imgcodecs
LDFLAGS = -L$(CUDA_HOME)/lib64 -L/opt/tools/opencv-3/lib
LDFLAGS_DEBUG = -L$(CUDA_HOME)/lib64 -L/opt/tools/opencv-3/lib
# define any directories containing header files other than /usr/include
#
INCLUDES= -I../include -I$(CUDA_HOME)/include -I/opt/tools/opencv-3/include 


# define the C source files
SRCS =  DNLMFilter.cpp ParallelDNLM.cpp

# define the C object files
#
# This uses Suffix Replacement within a macro:
#   $(name:string1=string2)
#         For each word in 'name' replace 'string1' with 'string2'
# Below we are replacing the suffix .c of all words in the macro SRCS
# with the .o suffix
#
OBJS = $(SRCS:.c=.o)

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

debug: $(OBJS) 
	   $(CC) $(OBJS) $(CFLAGS_DEBUG) $(INCLUDES)  -o $(MAIN) $(LDFLAGS_DEBUG) $(LIBS)



# this is a suffix replacement rule for building .o's from .c's
# it uses automatic variables $<: the name of the prerequisite of
# the rule(a .c file) and $@: the name of the target of the rule (a .o file)
# (see the gnu make manual section about automatic variables)
.c.o:
	 $(CC) -c $(CFLAGS) $(INCLUDES) $<



clean:
	rm $(MAIN) $(OBJS)

