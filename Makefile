trees:
	gcc -fPIC -shared -o trees.o -fopenmp trees.c

clean:
	rm trees.o
