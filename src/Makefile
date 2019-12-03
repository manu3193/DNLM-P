# define the C compiler to use
CC = nvcc 
PGI = pgc++
# define any compile-time flags
CFLAGS = -ccbin=pgc++ -Xcompiler " -std=c++11 -Minfo=accel  -ta=tesla:nordc -ta=tesla:cc35 -ta=tesla:lineinfo -Mcuda -O3" -O3 -cudart static  
LIBS = -lnppc_static -lnppial_static -lnppitc_static -lnppidei_static -lnppif_static -lnppisu_static -lnppist_static -lnpps_static -lculibos -lopencv_core -lopencv_highgui -lopencv_imgcodecs
LDFLAGS = -L$(CUDA_HOME)/lib64 -L/root/Manuel/opencv/lib64
# define any directories containing header files other than /usr/include
#
INCLUDES= -I../include -I$(CUDA_HOME)/include -I/root/Manuel/opencv/include 
# define the C source files
SRCS =  ParallelDNLM.cpp  DNLMFilter_OpenACC.cpp 
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
MAIN = nlmfilter
#
# The following part of the makefile is generic; it can be used to
# build any executable just by changing the definitions above and by
# deleting dependencies appended to the file from 'make depend'
#
.PHONY: clean
$(MAIN): $(OBJS) 
	$(CC) $(OBJS) $(CFLAGS) $(INCLUDES)  -o $@ $(LDFLAGS) $(LIBS)
#this is a suffix replacement rule for building .o's from .c's
# it uses automatic variables $<: the name of the prerequisite of
# the rule(a .c file) and $@: the name of the target of the rule (a .o file)
# (see the gnu make manual section about automatic variables)
.c.o:
	pgc++ -c $(PGFLAGS) $(INCLUDES) $<
clean:
