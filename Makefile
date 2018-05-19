all:
	make bfsMain
	make prMain

bfsMain:
	nvcc -ccbin clang-3.8 -I/usr/lib/mpich/include/ -lmpi main.cu -o bfsProgram

prMain:
	nvcc -ccbin clang-3.8 -I/usr/lib/mpich/include/ -lmpi mainPR.cu -o prProgram

prMainG5k:
	nvcc -I/usr/lib/x86_64-linux-gnu/openmpi/include -lmpi mainPR.cu -o prProgram

bfsMainG5k:
	nvcc -I/usr/lib/x86_64-linux-gnu/openmpi/include -lmpi main.cu -o bfsProgram
