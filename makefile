CC=gcc
CFLAGS=-g -Wall

all: generate examine

generate:
	$(CC) $(CFLAGS) src/generator.c -o generate
	./generate datafile.txt 15000000

examine:
	$(CC) $(CFLAGS) src/examine.c -o examine
	./examine -1 -1 datafile.txt -1 -1
