#
# 'make'        build executable file 'mycc'
# 'make clean'  removes all .o and executable files
#

IPP_LIBS_PATH = $(IPPROOT)/lib/intel64
# define the C compiler to use
CC = icpc
# define any compile-time flags
CFLAGS = -g -qopenmp -O3  -fma -align -finline-functions
LIBS = -lippi -lippcv -lippcc -lipps -lippvm  -lippcore `pkg-config --cflags --libs opencv`

LDFLAGS =-g3 -Wall #-qopt-report=2 -qopt-report-phase=vec
# define any directories containing header files other than /usr/include
#
INCLUDES= -I../include -I$(IPPROOT)/include 


# define the C source files
SRCS =  NoAdaptiveUSM.cpp DNLMFilter.cpp ParallelDNLM.cpp

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


# this is a suffix replacement rule for building .o's from .c's
# it uses automatic variables $<: the name of the prerequisite of
# the rule(a .c file) and $@: the name of the target of the rule (a .o file)
# (see the gnu make manual section about automatic variables)
.c.o:
	 $(CC) -c $(CFLAGS) $(INCLUDES) $<

clean:
	rm $(MAIN) $(OBJS)

