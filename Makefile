all:
	nvcc -ccbin clang-3.8 -I/usr/lib/mpich/include/ -lmpi bfs/bfsSingle.cu -o program
	nvcc -ccbin clang-3.8 -I/usr/lib/mpich/include/ -lmpi main.cu -o main

bfsMain:
	nvcc -ccbin clang-3.8 -I/usr/lib/mpich/include/ -lmpi main.cu -o main

prMain:
	nvcc -ccbin clang-3.8 -I/usr/lib/mpich/include/ -lmpi mainPR.cu -o mainPR

prMainG5k:
	nvcc -I/usr/lib/x86_64-linux-gnu/openmpi/include -lmpi mainPR.cu -o mainPR

bfsMainG5k:
	nvcc -I/usr/lib/x86_64-linux-gnu/openmpi/include -lmpi main.cu -o main
