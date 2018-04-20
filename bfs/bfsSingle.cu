#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

    int verticesPerProc;

    int* d_vertices;
    int* d_edges;
    int* d_visited;
    int* d_oldFrontier; //memcpy in the iteration loop
    int* d_newFrontier;
    int* d_levels;
    int* d_currentLevel;
    int* d_num_nodes;
    int* d_num_edges;
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

        //If vertices are odd, place last vertex in highest rank
        //if((num_nodes / world_size) % 2 != 0 && world_rank == (world_size -1)){
        //    verticesPerProc = (num_nodes / world_size) + 1;
        //    printf("vertices per proc EVEN OR ODD: %d\n", (num_nodes / world_size) % 2);
        //}
        //else{
        //    verticesPerProc = (num_nodes / world_size);
        //}

        //printf("vertices per proc: %d\n", verticesPerProc);

        cudaMalloc((void**)&d_num_nodes, sizeof(int));
        cudaMemcpy(d_num_nodes, &num_nodes, sizeof(int), cudaMemcpyHostToDevice);

        cudaMalloc((void**)&d_num_edges, sizeof(int));
        cudaMemcpy(d_num_edges, &num_edges, sizeof(int), cudaMemcpyHostToDevice);

        cudaMalloc((void**)&d_done, sizeof(int));
}

void distributedBFS(int* vertices, int* edges, int num_nodes, int num_edges, int world_rank, int world_size, int source){

    int* globalLevels;
    int* levels = (int *)malloc(num_nodes * sizeof(int));
    int* visited = (int *)malloc(num_nodes * sizeof(int));
    int* oldFrontier = (int *)malloc(num_nodes * sizeof(int));
    int* newFrontier = (int *)malloc(num_nodes * sizeof(int));

    int* globalOldFrontier = (int *)malloc(num_nodes * sizeof(int));
    int* globalNewFrontier = (int *)malloc(num_nodes * sizeof(int));

    int currentLevel = 0;

    int i;
    for(i=0; i < num_nodes; i++){
        levels[i] = -1;
        visited[i] = 0;
        //oldFrontier[i] = 0;
        //newFrontier[i] = 0;
        globalOldFrontier[i] = 0;
        globalNewFrontier[i] = 0;
    }

    //set the source value to 1
    //oldFrontier[source] = 1;
    globalOldFrontier[source] = 1;
    visited[source] = 1;

    //sentToGPU
    sendToGPU(vertices, edges, visited, newFrontier, levels, num_nodes, num_edges, world_size, world_rank);

    //Allocate sub arrays for MPI
    int* subOldFrontier = (int *)malloc((num_nodes/world_size) * sizeof(int));
    int* subNewFrontier = (int *)malloc((num_nodes/world_size) * sizeof(int));

    for(i=0; i < num_nodes/world_size; i++){
        subOldFrontier[i] = 0;
        subNewFrontier[i] = 0;
    }

    int threads = 1024;
    int blocks = 8;

    int done;
    int globalDone;

    int iterations = 0;
    printf("world_rank: %d\n", world_rank);

    do {
        if(iterations == 0 && world_rank == 0){
            iterations++;
            subOldFrontier[source] = 1;
            done = 1;

            cudaMemcpy(d_done, &done, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_currentLevel, &currentLevel, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_oldFrontier, subOldFrontier, sizeof(int) * (num_nodes/world_size), cudaMemcpyHostToDevice);

            CUDA_BFS_KERNEL <<<blocks, threads >>>(d_vertices, d_edges, d_oldFrontier, d_newFrontier,
                                            d_visited, d_levels, d_currentLevel, d_done, d_num_nodes, d_num_edges);

            cudaMemcpy(subNewFrontier, d_newFrontier, sizeof(int) * num_nodes/world_size, cudaMemcpyDeviceToHost);
            cudaMemcpy(&done, d_done , sizeof(int), cudaMemcpyDeviceToHost);

            printf("Done after gpu: %d\n", done);

            if(world_rank == 0){
                for(i=0; i < num_nodes/world_size; i++){
                    if(subNewFrontier[i] == 1)
                    printf("Process: %d , subNewFrontier[%d]: %d\n", world_rank, i, subNewFrontier[i]);
                }
            }

            //Reset old Frontier and copy new values from the new frontier to the old frontier for next iteration
            for(i=0; i < num_nodes/world_size; i++){
                subOldFrontier[i] = 0;
                if(subNewFrontier[i] == 1){
                    subOldFrontier[i] = 1;
                    subNewFrontier[i] = 0;
                }
            }

            currentLevel++;

            if(world_rank == 0){
                for(i=0; i < num_nodes/world_size; i++){
                    if(subOldFrontier[i] == 1)
                        printf("Process: %d , subOldFrontier[%d]: %d\n", world_rank, i, subOldFrontier[i]);
                }
            }

        }

        MPI_Gather(subOldFrontier, num_nodes/world_size, globalFrontier, sub_avgs, num_nodes/world_size, MPI_INT, 0, MPI_COMM_WORLD);
        //MPI_Allreduce(subOldFrontier, globalOldFrontier, num_nodes, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);

        for(i=0; i < num_nodes; i++){
            if(globalOldFrontier[i] == 1){
                subOldFrontier[i] = 1;
                globalOldFrontier[i] = 0;
            }
        }


        if(iterations > 0){
            MPI_Bcast(&currentLevel, 1, MPI_INT, 0, MPI_COMM_WORLD);

            iterations++;
            done = 1;

            cudaMemcpy(d_done, &done, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_currentLevel, &currentLevel, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_oldFrontier, subOldFrontier, sizeof(int) * num_nodes, cudaMemcpyHostToDevice);

            printf("\n");
            printf("ITERATION: %d\n", iterations);
            printf("NEW KERNEL CALLED\n");
            printf("\n");
            CUDA_BFS_KERNEL <<<blocks, threads >>>(d_vertices, d_edges, d_oldFrontier, d_newFrontier,
                                                   d_visited, d_levels, d_currentLevel, d_done, d_num_nodes, d_num_edges);

            cudaMemcpy(subNewFrontier, d_newFrontier, sizeof(int) * num_nodes, cudaMemcpyDeviceToHost);
            cudaMemcpy(&done, d_done , sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(levels, d_levels, sizeof(int) * num_nodes, cudaMemcpyDeviceToHost);

            MPI_Allreduce(subNewFrontier, globalNewFrontier, num_nodes, MPI_INT, MPI_LOR, MPI_COMM_WORLD);

            MPI_Allreduce(&done, &globalDone, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
            printf("PROCESS %d : Done: %d\n", world_rank, done);
            printf("GlobalDone: %d\n", globalDone);

            for(i=0; i < num_nodes; i++){
                subOldFrontier[i] = 0;
            }

            if(world_rank == 0){
                for(i=0; i < num_nodes; i++){
                    globalOldFrontier[i] = 0;
                    if(globalNewFrontier[i] == 1){
                        globalOldFrontier[i] = 1;
                        globalNewFrontier[i] = 0;
                    }
                }
            }
            currentLevel++;

            if(world_rank == 0){
                for(i=0; i < num_nodes; i++){
                    if(globalOldFrontier[i] == 1)
                        printf("Process: %d , globalOldFrontier[%d]: %d\n", world_rank, i, globalOldFrontier[i]);
                }
            }
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