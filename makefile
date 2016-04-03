CC=gcc
CFLAGS=-g -Wall
EXE=generator examine
all: generate examine

generate: generator.o
	$(CC) $(CFLAGS) generator.o -o generator
	./generator datafile.txt 15000000

examine: examine.o
	$(CC) $(CFLAGS) -fopenmp src/examine.c -o examine
	./examine -1 -1 datafile.txt -1 -1

generator.o: src/generator.c
	$(CC) -c $(CFLAGS) $< -o $@

examine.o: src/examine.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm *.o datafile.txt $(EXE)
