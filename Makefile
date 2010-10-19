PROGS = solve_cyclic
CC = gcc
CFLAGS = -g -Wall -O3
LDLIBS = -lcfitsio -lfftw3f_threads -lfftw3f -lm
all: $(PROGS)
clean:
	rm -rf $(PROGS) *.o
install: $(PROGS)
	cp -f $(PROGS) $(LOCAL)/bin
solve_cyclic: solve_cyclic.o cyclic_utils.o 
