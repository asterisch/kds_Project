CC=mpicc
CFLAGS=-g -Wall
RUNARGS=-np 13
EXE=generator examine
all: generate examine

generate: generator.o
	$(CC) $(CFLAGS) generator.o -o generator
	mpirun generator datafile.txt 15000000

examine: examine.o
	$(CC) $(CFLAGS) -fopenmp src/examine.c  -o examine
	mpirun $(RUNARGS) examine -1 -1 datafile.txt -1 -1

generator.o: src/generator.c
	$(CC) -c $(CFLAGS) $< -o $@

examine.o: src/examine.c
	$(CC) $(CFLAGS) -fopenmp -c $< -o $@

clean:
	rm *.o datafile.txt $(EXE)
