all:
	nvcc -ccbin clang-3.8 -I/usr/lib/mpich/include/ -lmpi bfs/bfsSingle.cu -o program
	nvcc -ccbin clang-3.8 -I/usr/lib/mpich/include/ -lmpi main.cu -o main

bfs:
	nvcc -ccbin clang-3.8 -I/usr/lib/mpich/include/ -lmpi bfs/bfsSingle.cu -o program

mai:
	nvcc -ccbin clang-3.8 -I/usr/lib/mpich/include/ -lmpi main.cu -o main
