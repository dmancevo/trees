trees:
	gcc -o trees.o -fopenmp trees.c

run:
	gcc -o trees.o -fopenmp trees.c
	./trees.o
	rm trees.o


clean:
	rm trees.o
