all:Main

Main:Main.o Examples.o
	nvcc -o Main Main.o Examples.o -lcurand 
	rm *.o
Main.o:Main.cu
	nvcc -c Main.cu 
Examples.o:Examples.cu
	nvcc -c Examples.cu
