ifeq ($(COMP),)
COMP=gnu
endif

COMMONDIR=../common

ifeq ($(COMP),pgi)
CXX=pgCC
CC=pgcc
CCFLAGS=-g -O3 -fopenmp -I$(COMMONDIR)
LDFLAGS=
LIBS=-lpng
endif

ifeq ($(COMP),gnu)
CXX=g++
CC=gcc
CCFLAGS=-O3 -fopenmp -Wall -Wextra -Wshadow -pedantic -I$(COMMONDIR) -I/opt/rocm/include
LDFLAGS=
LIBS=-lpng
endif

ifeq ($(COMP),intel)
CXX=icpx
CC=icx
CCFLAGS=-g -O3 -fopenmp -I$(COMMONDIR)
LDFLAGS=
LIBS=-lpng
endif

ifeq ($(COMP),lumi)
LIBPNG_DIR=/appl/lumi/SW/LUMI-22.12/C/EB/libpng/1.6.38-cpeCray-22.12
CXX=CC
CC=cc
CCFLAGS=-O3 -fopenmp -Wall -xhip -I$(COMMONDIR) -I$(LIBPNG_DIR)/include -DNDEBUG
LDFLAGS=-L$(LIBPNG_DIR)/lib -Wl,-rpath=$(LIBPNG_DIR)/lib
LIBS=-lpng
endif

EXE=heat_serial
OBJS=main.o core.o setup.o utilities.o io.o
OBJS_PNG=$(COMMONDIR)/pngwriter.o


all: $(EXE)


core.o: core.cpp heat.h
utilities.o: utilities.cpp heat.h
setup.o: setup.cpp heat.h
io.o: io.cpp heat.h
main.o: main.cpp heat.h

$(OBJS_PNG): C_COMPILER := $(CC)
$(OBJS): C_COMPILER := $(CXX)

$(EXE): $(OBJS) $(OBJS_PNG)
	$(CXX) $(CCFLAGS) $(OBJS) $(OBJS_PNG) -o $@ $(LDFLAGS) $(LIBS)

just_do_it: just_do_it.cpp
	$(CXX) $(CCFLAGS) just_do_it.cpp -o $@ $(LDFLAGS) $(LIBS)

%.o: %.cpp
	$(CXX) $(CCFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@

.PHONY: clean
clean:
	-/bin/rm -f $(EXE) a.out *.o *.png *~
