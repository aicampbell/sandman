#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

    int elementsPerProc;

    int* d_vertices;
    int* d_edges;
    int* d_visited;
    int* d_oldFrontier; //memcpy in the iteration loop
    int* d_newFrontier;
    int* d_levels;
    int* d_currentLevel;
    int* d_num_nodes;
    int* d_num_edges;
    int* d_elementsPerProc;
    int* d_done;

    int world_rank;
    int world_size;

__global__ void CUDA_BFS_KERNEL(int *d_vertices, int *d_edges, int* d_oldFrontier, int* d_newFrontier, int* d_visited,
     int* d_levels, int *d_currentLevel, int *d_done, int *d_num_nodes, int *d_num_edges){

    int i;

    //Set the thread id
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if(id > *d_num_nodes)
        *d_done = 1;

    d_newFrontier[id] = 0;
    __syncthreads();

    if (d_oldFrontier[id] == 1 && d_visited[id] == 1){
        printf("Node order: %d \n", id); //This printf gives the order of vertices in BFS

        d_levels[id] = *d_currentLevel; //set the level of the current node

        int start = d_vertices[id];
        int end = d_vertices[id + 1];

        for(i = start; i < end; i++){
            int nid = d_edges[i];
            //printf("GPU Nid: %d\n", d_edges[i]);
            if(d_visited[nid] == false){
                d_visited[nid] = 1;
                d_newFrontier[nid] = 1;
                *d_done = 0;
            }
        }
    }
    __syncthreads();
}

void sendToGPU(int* vertices, int* edges, int* visited, int* newFrontier, int* levels, int num_nodes, int num_edges, int world_size,
                int world_rank){

    //COPY to GPU

        cudaMalloc((void**)&d_vertices, sizeof(int) * num_nodes);
        cudaMemcpy(d_vertices, vertices, sizeof(int) * num_nodes, cudaMemcpyHostToDevice);

        cudaMalloc((void**)&d_edges, sizeof(int) * num_edges);
        cudaMemcpy(d_edges, edges, sizeof(int) * num_edges, cudaMemcpyHostToDevice);

        cudaMalloc((void**)&d_visited, sizeof(int) * num_nodes);
        cudaMemcpy(d_visited, visited, sizeof(int) * num_nodes, cudaMemcpyHostToDevice);

        cudaMalloc((void**)&d_oldFrontier, sizeof(int) * num_nodes);

        cudaMalloc((void**)&d_newFrontier, sizeof(int) * num_nodes);
        cudaMemcpy(d_newFrontier, newFrontier, sizeof(int) * num_nodes, cudaMemcpyHostToDevice);

        cudaMalloc((void**)&d_levels, sizeof(int) * num_nodes);
        cudaMemcpy(d_levels, levels, sizeof(int) * num_nodes, cudaMemcpyHostToDevice);

        cudaMalloc((void**)&d_currentLevel, sizeof(int));

        if((num_edges / world_size) % 2 != 0 && world_rank == (world_size -1)){
            elementsPerProc = (num_edges / world_size) + 1;
            printf("elements per proc EVEN OR ODD: %d\n", (num_edges / world_size) % 2);
        }
        else{
            elementsPerProc = (num_edges / world_size);
        }

        printf("elements per proc: %d\n", elementsPerProc);

        cudaMalloc((void**)&d_num_nodes, sizeof(int));
        cudaMemcpy(d_num_nodes, &num_nodes, sizeof(int), cudaMemcpyHostToDevice);

        cudaMalloc((void**)&d_num_edges, sizeof(int));
        cudaMemcpy(d_num_edges, &num_edges, sizeof(int), cudaMemcpyHostToDevice);

        cudaMalloc((void**)&d_elementsPerProc, sizeof(int));
        cudaMemcpy(d_elementsPerProc, &elementsPerProc, sizeof(int), cudaMemcpyHostToDevice);

        cudaMalloc((void**)&d_done, sizeof(int));
}

void distributedBFS(int* vertices, int* edges, int num_nodes, int num_edges, int world_rank, int world_size, int source){

    int* globalLevels;
    int* levels = (int *)malloc(num_nodes * sizeof(int));
    int* visited = (int *)malloc(num_nodes * sizeof(int));
    int* oldFrontier = (int *)malloc(num_nodes * sizeof(int));
    int* newFrontier = (int *)malloc(num_nodes * sizeof(int));
    int currentLevel = 0;

    int i;
    for(i=0; i < num_nodes; i++){
        levels[i] = -1;
        visited[i] = 0;
        oldFrontier[i] = 0;
        newFrontier[i] = 0;
    }

    //set the source value to 1
    oldFrontier[source] = 1;
    visited[source] = 1;

    //sentToGPU
    sendToGPU(vertices, edges, visited, newFrontier, levels, num_nodes, num_edges, world_size, world_rank);

    //Allocate sub arrays for MPI
    int* subOldFrontier = (int *)malloc((elementsPerProc + 1) * sizeof(int));
    int* subNewFrontier = (int *)malloc((elementsPerProc + 1) * sizeof(int));

    for(i=0; i < elementsPerProc; i++){
       subOldFrontier[i] = 0;
       subNewFrontier[i] = 0;
    }

    int threads = 1024;
    int blocks = 8;

    int done;
    int globalDone;

    int* globalOldFrontier = (int *)malloc(num_nodes * sizeof(int));
    int* globalNewFrontier = (int *)malloc(num_nodes * sizeof(int));

    int iterations = 0;
    printf("world_rank: %d\n", world_rank);

    do {
        if(iterations == 0 && world_rank == 0){
            iterations++;
            done = 1;

            cudaMemcpy(d_done, &done, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_currentLevel, &currentLevel, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_oldFrontier, oldFrontier, sizeof(int) * num_nodes, cudaMemcpyHostToDevice);

            CUDA_BFS_KERNEL <<<blocks, threads >>>(d_vertices, d_edges, d_oldFrontier, d_newFrontier,
                                            d_visited, d_levels, d_currentLevel, d_done, d_num_nodes, d_num_edges);

            cudaMemcpy(newFrontier, d_newFrontier, sizeof(int) * num_nodes, cudaMemcpyDeviceToHost);
            cudaMemcpy(&done, d_done , sizeof(int), cudaMemcpyDeviceToHost);

            printf("Done after gpu: %d", done);

           for(i=0; i < num_nodes; i++){
                oldFrontier[i] = 0;
                if(newFrontier[i] == 1){
                    oldFrontier[i] = 1;
                    newFrontier[i] = 0;
                }
           }

            currentLevel++;
        }

        MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(oldFrontier, num_nodes, MPI_INT, 0, MPI_COMM_WORLD);

        if(iterations > 0){
            MPI_Bcast(&currentLevel, 1, MPI_INT, 0, MPI_COMM_WORLD);

            iterations++;
            done = 1;

            cudaMemcpy(d_done, &done, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_currentLevel, &currentLevel, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_oldFrontier, oldFrontier, sizeof(int) * num_nodes, cudaMemcpyHostToDevice);

            printf("\n");
            printf("\n");
            printf("NEW KERNEL CALLED\n");
            printf("\n");
            printf("\n");
            CUDA_BFS_KERNEL <<<blocks, threads >>>(d_vertices, d_edges, d_oldFrontier, d_newFrontier,
                                                   d_visited, d_levels, d_currentLevel, d_done, d_num_nodes, d_num_edges);

            cudaMemcpy(newFrontier, d_newFrontier, sizeof(int) * num_nodes, cudaMemcpyDeviceToHost);
            cudaMemcpy(&done, d_done , sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(levels, d_levels, sizeof(int) * num_nodes, cudaMemcpyDeviceToHost);

            MPI_Allreduce(newFrontier, globalNewFrontier, num_nodes, MPI_INT, MPI_LOR, MPI_COMM_WORLD);

            MPI_Allreduce(&done, &globalDone, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
            printf("PROCESS %d : Done: %d\n", world_rank, done);
            printf("GlobalDone: %d\n", globalDone);

            for(i=0; i < num_nodes; i++){
                oldFrontier[i] = 0;
                if(newFrontier[i] == 1){
                    oldFrontier[i] = 1;
                    newFrontier[i] = 0;
                }
            }
            currentLevel++;
        }

    } while (globalDone == 0);

    if(world_rank == 0){
        printf("Number of times the kernel is called : %d \n", iterations);
        globalLevels = (int *)malloc(num_nodes * sizeof(int));
    }

    MPI_Reduce(levels, globalLevels, num_nodes, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    if(world_rank == 0){
        printf("\nLevel:\n");
        for (i = 0; i < num_nodes; i++)
            printf("node %d level: %d\n", i, globalLevels[i]);
        printf("\n");
    }
}