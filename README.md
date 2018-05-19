# sandman

To Build:
   make all

To run:
   mpirun -np <num_partitions> ./bfsProgram <graph.mtx>
   mpirun -np <num_partitions> ./prProgram <graph.mtx>

Need to pipe output to a file otherwise it will print to screen