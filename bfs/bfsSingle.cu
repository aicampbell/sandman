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
    int* d_done;
    int* d_process_id;
    int* d_elements_per_process;
    int* d_edgeOffset;

    int world_rank;
    int world_size;

__global__ void CUDA_BFS_KERNEL(int *d_vertices, int *d_edges, int* d_oldFrontier, int* d_newFrontier, int* d_visited,
     int* d_levels, int *d_currentLevel, int *d_done, int *d_num_nodes, int *d_num_edges, int *d_process_id,
     int *d_elements_per_process, int *d_edgeOffset){

    int i;

    //Set the thread id
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = *d_process_id * *d_elements_per_process;

    if(d_oldFrontier[id] == 1 && d_visited[id + offset] == 1 && *d_process_id == 1){
        //printf("Thread ID: %d\n", id);
    }

    if(id > *d_num_nodes)
        *d_done = 1;

    d_newFrontier[id + offset] = 0;
    __syncthreads();

    if (d_oldFrontier[id] == 1 && d_visited[id + offset] == 1){
        //printf("Node order: %d \n", id + offset); //This printf gives the order of vertices in BFS
        d_levels[id + offset] = *d_currentLevel; //set the level of the current node

        int start = d_vertices[id + offset];
        int end = d_vertices[id + 1 + offset];

        //printf("start: %d\n", start);
        //printf("end: %d\n", end);

        for(i = start; i < end; i++){
            int nid = d_edges[i - *d_edgeOffset];
            //printf("GPU Nid: %d\n", nid);
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
                int world_rank, int edgeOffset){

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
        if((num_nodes / world_size) % 2 != 0 && world_rank == (world_size -1)){
            elementsPerProc = (num_nodes / world_size) + 1;
            printf("vertices per proc EVEN OR ODD: %d\n", (num_nodes / world_size) % 2);
        }
        else{
            elementsPerProc = (num_nodes / world_size);
        }

        printf("Process %d elements per proc: %d\n", world_rank, elementsPerProc);

        cudaMalloc((void**)&d_elements_per_process, sizeof(int));
        cudaMemcpy(d_elements_per_process, &elementsPerProc, sizeof(int), cudaMemcpyHostToDevice);

        cudaMalloc((void**)&d_process_id, sizeof(int));
        cudaMemcpy(d_process_id, &world_rank, sizeof(int), cudaMemcpyHostToDevice);

        cudaMalloc((void**)&d_num_nodes, sizeof(int));
        cudaMemcpy(d_num_nodes, &num_nodes, sizeof(int), cudaMemcpyHostToDevice);

        cudaMalloc((void**)&d_num_edges, sizeof(int));
        cudaMemcpy(d_num_edges, &num_edges, sizeof(int), cudaMemcpyHostToDevice);

        cudaMalloc((void**)&d_edgeOffset, sizeof(int));
        cudaMemcpy(d_edgeOffset, &edgeOffset, sizeof(int), cudaMemcpyHostToDevice);

        cudaMalloc((void**)&d_done, sizeof(int));
}

void distributedBFS(int* vertices, int* edges, int num_nodes, int num_edges, int world_rank, int world_size, int source, int edgeOffset){

    int* globalLevels = (int *)malloc(num_nodes * sizeof(int));
    int offset;
    int* levels = (int *)malloc(num_nodes * sizeof(int));
    int* visited = (int *)malloc(num_nodes * sizeof(int));
    int* oldFrontier = (int *)malloc(num_nodes * sizeof(int));
    int* newFrontier = (int *)malloc(num_nodes * sizeof(int));

    int* globalOldFrontier = (int *)malloc(num_nodes * sizeof(int));
    int* globalNewFrontier = (int *)malloc(num_nodes * sizeof(int));
    int* globalVisited = (int *)malloc(num_nodes * sizeof(int));

    int currentLevel = 0;

    int i;
    for(i=0; i < num_nodes; i++){
        levels[i] = -1;
        visited[i] = 0;
        globalVisited[i] = 0;
        globalOldFrontier[i] = 0;
        globalNewFrontier[i] = 0;
    }

    //set the source value to 1
    globalOldFrontier[source] = 1;
    visited[source] = 1;

    //sentToGPU
    sendToGPU(vertices, edges, visited, newFrontier, levels, num_nodes, num_edges, world_size, world_rank, edgeOffset);
    offset = world_rank * elementsPerProc;
    printf("Process %d offset: %d\n", world_rank, offset);
    printf("Process %d edge offset: %d\n", world_rank, edgeOffset);

    //Allocate sub arrays for MPI
    int* subOldFrontier = (int *)malloc(elementsPerProc * sizeof(int));
    int* subNewFrontier = (int *)malloc(num_nodes * sizeof(int));

    for(i=0; i < elementsPerProc; i++){
        subOldFrontier[i] = 0;
        subNewFrontier[i] = 0;
    }

    int threads = 1024;
    int blocks = num_nodes/1024;

    int done;
    int globalDone;

    int iterations = 0;

    do {
        if(iterations == 0 && world_rank == 0){
        iterations++;

        cudaMemcpy(d_done, &done, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_currentLevel, &currentLevel, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_oldFrontier, globalOldFrontier, sizeof(int) * num_nodes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_visited, visited, sizeof(int) * num_nodes, cudaMemcpyHostToDevice);

        CUDA_BFS_KERNEL <<<blocks, threads >>>(d_vertices, d_edges, d_oldFrontier, d_newFrontier,
                                           d_visited, d_levels, d_currentLevel, d_done, d_num_nodes, d_num_edges,
                                           d_process_id, d_elements_per_process, d_edgeOffset);

        cudaMemcpy(&done, d_done , sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(globalNewFrontier, d_newFrontier, sizeof(int) * num_nodes, cudaMemcpyDeviceToHost);
        cudaMemcpy(visited, d_visited, sizeof(int) * num_nodes, cudaMemcpyDeviceToHost);

        //Reset the values:
        for(i = 0; i < num_nodes; i++){
            globalOldFrontier[i] = 0;
            if(globalNewFrontier[i] == 1){
                globalOldFrontier[i] = 1;
                globalNewFrontier[i] = 0;
            }
        }

        currentLevel++;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Scatter(globalOldFrontier, elementsPerProc, MPI_INT, subOldFrontier, elementsPerProc, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Allreduce(visited, globalVisited, num_nodes, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        //for(i = 0; i < num_nodes; i++){
        //    globalOldFrontier[i] = 0;
        //    if(globalNewFrontier[i] == 1){
        //        globalOldFrontier[i] = 1;
        //        globalNewFrontier[i] = 0;
        //    }
        //}

        if(iterations > 0){

        MPI_Bcast(&currentLevel, 1, MPI_INT, 0, MPI_COMM_WORLD);

         if(world_rank == 1){
            for(i = 0; i < elementsPerProc; i++){
                if(subOldFrontier[i] == 1){
                    printf("subOldFrontier[%d] = %d\n", i + offset, subOldFrontier[i]);
                }
            }
         }

        iterations++;
        done = 1;

        cudaMemcpy(d_done, &done, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_currentLevel, &currentLevel, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_oldFrontier, subOldFrontier, sizeof(int) * elementsPerProc, cudaMemcpyHostToDevice);
        cudaMemcpy(d_visited, globalVisited, sizeof(int) * num_nodes, cudaMemcpyHostToDevice);


        CUDA_BFS_KERNEL <<<blocks, threads >>>(d_vertices, d_edges, d_oldFrontier, d_newFrontier,
                            d_visited, d_levels, d_currentLevel, d_done, d_num_nodes, d_num_edges,
                            d_process_id, d_elements_per_process, d_edgeOffset);

        cudaMemcpy(&done, d_done , sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(subNewFrontier, d_newFrontier, sizeof(int) * num_nodes, cudaMemcpyDeviceToHost);
        cudaMemcpy(visited, d_visited, sizeof(int) * num_nodes, cudaMemcpyDeviceToHost);
        cudaMemcpy(levels, d_levels, sizeof(int) * num_nodes, cudaMemcpyDeviceToHost);

        MPI_Allreduce(&done, &globalDone, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
        printf("GlobalDone: %d\n", globalDone);
        printf("Iteration: %d\n", iterations);

        int count = 0;
        for(i=0; i < elementsPerProc; i++){
            if(subOldFrontier[i] == 1){
                count++;
            }
        }
        printf("PRocess %d: new frontier count: %d\n", world_rank, count);


        for(i=0; i < elementsPerProc; i++){
            subOldFrontier[i] = 0;
        }

        for(i=offset; i < (world_rank + 1) * elementsPerProc; i++){
            if(subNewFrontier[i] == 1){
                subOldFrontier[i - offset] = 1;
                subNewFrontier[i] = 0;
            }
        }

        MPI_Gather(subOldFrontier, elementsPerProc, MPI_INT, globalOldFrontier, elementsPerProc, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Allreduce(visited, globalVisited, num_nodes, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        currentLevel++;
        }

        if(world_rank == 0){
            for(i = 0; i < num_nodes; i++){
                if(globalOldFrontier[i] == 1){
                    //printf("globalOldFrontier[%d] = %d\n", i, globalOldFrontier[i]);
                }
            }
        }

    } while (globalDone == 0);

    if(world_rank == 0){
        printf("Number of times the kernel is called : %d \n", iterations);

    }


    MPI_Reduce(levels, globalLevels, num_nodes, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    if(world_rank == 0){
        for(i=0; i < num_nodes; i++){

        if(globalLevels[i] == 0 && i != source){
            globalLevels[i] = -1;
        }
    }
    }

    if(world_rank == 0){
       printf("num_nodes: %d\n", num_nodes);
       printf("\nLevel:\n");
       for (i = 0; i < num_nodes; i++)
           printf("node %d level: %d\n", i, globalLevels[i]);
       printf("\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
}