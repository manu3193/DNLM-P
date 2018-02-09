#
# 'make'        build executable file 'mycc'
# 'make clean'  removes all .o and executable files
#

IPP_IW_ROOT = /home/mzumbado/lib/l_ippiw_p_2017.3.013/ippiw_2017_linux
IPP_LIBS_PATH = $(IPPROOT)/lib/intel64
IPP_IW_LIBS_PATH = $(IPP_IW_ROOT)/lib/intel64
# define the C compiler to use
CC = icc
# define any compile-time flags
CFLAGS = -qopenmp -O3 -xmic-avx512 -ipp=common  -fma -align -finline-functions
LIBS = $(IPP_IW_LIBS_PATH)/libipp_iw.a $(IPP_LIBS_PATH)/libippi.a $(IPP_LIBS_PATH)/libippcc.a $(IPP_LIBS_PATH)/libipps.a $(IPP_LIBS_PATH)/libippvm.a  $(IPP_LIBS_PATH)/libippcore.a

LDFLAGS =-g3 -Wall
# define any directories containing header files other than /usr/include
#
INCLUDES= -I$(IPPROOT)/include -I$(IPP_IW_ROOT)/include


# define the C source files
SRCS = hello.c

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
MAIN = hello

#
# The following part of the makefile is generic; it can be used to
# build any executable just by changing the definitions above and by
# deleting dependencies appended to the file from 'make depend'
#

.PHONY: clean

all:
	$(CC) $(OBJS) $(CFLAGS) $(INCLUDES) -o $(MAIN) $(LDFLAGS) $(LIBS)

# this is a suffix replacement rule for building .o's from .c's
# it uses automatic variables $<: the name of the prerequisite of
# the rule(a .c file) and $@: the name of the target of the rule (a .o file)
# (see the gnu make manual section about automatic variables)
.c.o: $(CC) $(CFLAGS) $(INCLUDES) -c $<  -o $@

clean:
	rm *.o *~ $(MAIN)


# DO NOT DELETE THIS LINE -- make depend needs it
