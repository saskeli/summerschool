ifeq ($(COMP),)
COMP=intel
endif

COMMONDIR=../../../heat/common

ifeq ($(COMP),intel)
CXX=icpc
CC=icc
CCFLAGS=-g -qopenmp -O3 -xHost -I$(COMMONDIR)
LDFLAGS=
LIBS=-lpng
endif

ifeq ($(COMP),gnu)
FC=gfortran
CC=gcc
FCFLAGS=-fopenmp -O3 -Wall
CCFLAGS=-fopenmp -O3 -Wall -I$(COMMONDIR)
LDFLAGS=
LIBS=-lpng
endif

ifeq ($(COMP),lumi)
LIBPNG_DIR=/appl/lumi/SW/LUMI-22.12/C/EB/libpng/1.6.38-cpeCray-22.12
CXX=CC
CC=cc
CCFLAGS=-O3 -fopenmp -Wall -I$(COMMONDIR) -I$(LIBPNG_DIR)/include
LDFLAGS=-L$(LIBPNG_DIR)/lib -Wl,-rpath=$(LIBPNG_DIR)/lib
LIBS=-lpng
endif

EXE=mandelbrot
OBJS=mandelbrot.o
OBJS_PNG=pngwriter.o

all: $(EXE)

$(COMMONDIR)/pngwriter.o: $(COMMONDIR)/pngwriter.c $(COMMONDIR)/pngwriter.h
mandelbrot.o: mandelbrot.cpp

pngwriter.o: pngwriter.c pngwriter.h
	$(CC) $(CCFLAGS) -c -o $@ $<

$(EXE): $(OBJS) $(OBJS_PNG)
	$(CC) $(CCFLAGS) $(OBJS) $(OBJS_PNG) -o $@ $(LDFLAGS) $(LIBS)

%.o: %.F90
	$(FC) $(FCFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CCFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@

.PHONY: clean
clean:
	-/bin/rm -f $(EXE) a.out *.o *.mod *.png *~ mandelbrot.png
