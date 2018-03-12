# sandman

To Build,
   make bfsMain

To run:
   mpirun -np <num_partitions> ./main <graph.mtx>

BFS Method:
Need to remove the BFS single method as it will only use distributed method, even with only 1 partiton.
Still use this file:
/bfs/bfsSingle.cu

Main:
Handles reading file, CSR, Partitions etc.
