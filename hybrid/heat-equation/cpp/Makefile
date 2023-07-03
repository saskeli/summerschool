ifeq ($(COMP),)
COMP=gnu
endif

COMMONDIR=../common

ifeq ($(COMP),gnu)
CXX=mpicxx
CC=mpicc
CCFLAGS=-g -fopenmp -O0 -march=native -Wall -I$(COMMONDIR) -DNDEBUG
LDFLAGS=
LIBS=-lpng
endif

ifeq ($(COMP),intel)
CXX=icpc
CC=icc
CCFLAGS=-g -O3 -xHost -I$(COMMONDIR) -DNDEBUG
LDFLAGS=
LIBS=-lpng
endif

ifeq ($(COMP),lumi)
LIBPNG_DIR=/appl/lumi/SW/LUMI-22.12/C/EB/libpng/1.6.38-cpeCray-22.12
CXX=CC
CC=cc
CCFLAGS=-O3 -fopenmp -Wall -I$(COMMONDIR) -I$(LIBPNG_DIR)/include -DNDEBUG
LDFLAGS=-L$(LIBPNG_DIR)/lib -Wl,-rpath=$(LIBPNG_DIR)/lib
LIBS=-lpng
endif

EXE=heat_omp
OBJS=main.o heat.o core.o setup.o io.o utilities.o
OBJS_PNG=$(COMMONDIR)/pngwriter.o


all: $(EXE)


utilities.o: utilities.cpp heat.hpp
heat.o: heat.cpp heat.hpp matrix.hpp
core.o: core.cpp heat.hpp
setup.o: setup.cpp heat.hpp
io.o: io.cpp heat.hpp matrix.hpp
main.o: main.cpp heat.hpp

$(OBJS_PNG): C_COMPILER := $(CC)
$(OBJS): C_COMPILER := $(CXX)

$(EXE): $(OBJS) $(OBJS_PNG)
	$(CXX) just_do_it.cpp $(CCFLAGS) $(OBJS) $(OBJS_PNG) -o $@ $(LDFLAGS) $(LIBS)

just_do_it: just_do_it.cpp $(OBJS_PNG)
	$(CXX) $(CCFLAGS) -std=c++20 -Wextra -Wshadow -pedantic just_do_it.cpp $(OBJS_PNG) -o $@ $(LDFLAGS) $(LIBS)

%.o: %.cpp
	$(CXX) $(CCFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@

.PHONY: clean
clean:
	-/bin/rm -f $(EXE) a.out *.o *.png *~
