#
# 'make'        build executable file 'mycc'
# 'make clean'  removes all .o and executable files
#

IPP_LIBS_PATH = $(IPPROOT)/lib/intel64
OPENCV3_HOME = /opt/tools/opencv-3

# define the C compiler to use
CC = icpc
# define any compile-time flags
CFLAGS_KNL = -std=c++11 -qopenmp -O3 -xMIC-AVX512  -fma -align -finline-functions 
CFLAGS_DEBUG = -g -std=c++11 -qopenmp -O0  -align -finline-functions 
CFLAGS_KNL_DEBUG = -g -std=c++11 -qopenmp -O0 -xMIC-AVX512  -align -finline-functions 
CFLAGS = -std=c++11  -qopenmp -O3  -xCORE-AVX2 -fma -align -finline-functions 

LIBS = -lippi -lippcv -lippcc -lipps -lippvm  -lippcore  -lopencv_core -lopencv_highgui -lopencv_imgcodecs 

LDFLAGS =-g3 -Wall -L$(IPP_LIBS_PATH) -L$(OPENCV3_HOME)/lib 
LDFLAGS_DEBUG =-g3 -Wall -L$IPP_LIBS_PATH -qopt-report=5  -qopt-report-phase=loop,vec,cg -qopt-report-embed
# define any directories containing header files other than /usr/include
#
INCLUDES= -I../include -I$(IPPROOT)/include -I$(OPENCV3_HOME)/include 


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
MAIN = nlmfilter_xeon_goossens
MAIN_KNL = nlmfilter_knl_goossens

#
# The following part of the makefile is generic; it can be used to
# build any executable just by changing the definitions above and by
# deleting dependencies appended to the file from 'make depend'
#

.PHONY: clean


$(MAIN): $(OBJS) 
	   $(CC) $(OBJS) $(CFLAGS) $(INCLUDES)  -o $@ $(LDFLAGS) $(LIBS)

knl: $(OBJS) 
	   $(CC) $(OBJS) $(CFLAGS_KNL) $(INCLUDES)  -o $(MAIN_KNL) $(LDFLAGS) $(LIBS)

debug: $(OBJS) 
	   $(CC) $(OBJS) $(CFLAGS_DEBUG) $(INCLUDES)  -o $(MAIN) $(LDFLAGS_DEBUG) $(LIBS)

knl-debug: $(OBJS) 
	   $(CC) $(OBJS) $(CFLAGS_KNL_DEBUG) $(INCLUDES)  -o $(MAIN_KNL) $(LDFLAGS_DEBUG) $(LIBS)


# this is a suffix replacement rule for building .o's from .c's
# it uses automatic variables $<: the name of the prerequisite of
# the rule(a .c file) and $@: the name of the target of the rule (a .o file)
# (see the gnu make manual section about automatic variables)
.c.o:
	 $(CC) -c $(CFLAGS) $(INCLUDES) $<



clean:
	rm $(MAIN) $(OBJS)

