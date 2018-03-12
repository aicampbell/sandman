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
        *d_done = 0;

    d_newFrontier[id] = 0;
    __syncthreads();

    if (d_oldFrontier[id] == 1 && d_visited[id] == 1){
        printf("Node order: %d \n", id); //This printf gives the order of vertices in BFS

        d_levels[id] = *d_currentLevel; //set the level of the current node

        int start = d_vertices[id];
        int end;
        if(id == 4){
            start = 7;
            end = *d_num_edges - 1;
        }else{
            end = d_vertices[id + 1];
        }

        printf("Id: %d -Start %d\n", id, start);
        printf("Id: %d - End %d\n", id, end);
        for(i = start; i < end; i++){
            int nid = d_edges[i];
            if(d_visited[nid] == false){
                d_visited[nid] = 1;
                d_newFrontier[nid] = 1;
                *d_done = 0;
            }
        }
    }
    __syncthreads();
}

void singleBFS(int NUM_NODES){

int i;

    int vertices[NUM_NODES];
    int levels[NUM_NODES];

    //0 == false
    //1 == true
    int visited[NUM_NODES];
    int oldFrontier[NUM_NODES];
    int newFrontier[NUM_NODES];

    int currentLevel = 0;

    int edges[NUM_NODES];

    for(i=0; i < NUM_NODES; i++){
        levels[i] = -1;
        visited[i] = 0;
        oldFrontier[i] = 0;
        newFrontier[i] = 0;
    }

    //set source
    int source = 0;

    oldFrontier[source] = 1;
    visited[source] = 1;

    //COPY to GPU
    int* d_vertices;
    cudaMalloc((void**)&d_vertices, sizeof(int) * NUM_NODES);
    cudaMemcpy(d_vertices, vertices, sizeof(int) * NUM_NODES, cudaMemcpyHostToDevice);

    int* d_edges;
    cudaMalloc((void**)&d_edges, sizeof(int) * NUM_NODES);
    cudaMemcpy(d_edges, edges, sizeof(int) * NUM_NODES, cudaMemcpyHostToDevice);

    int* d_visited;
    cudaMalloc((void**)&d_visited, sizeof(int) * NUM_NODES);
    cudaMemcpy(d_visited, visited, sizeof(int) * NUM_NODES, cudaMemcpyHostToDevice);

    int* d_oldFrontier; //memcpy in the iteration loop
    cudaMalloc((void**)&d_oldFrontier, sizeof(int) * NUM_NODES);

    int* d_newFrontier;
    cudaMalloc((void**)&d_newFrontier, sizeof(int) * NUM_NODES);
    cudaMemcpy(d_newFrontier, &newFrontier, sizeof(int) * NUM_NODES, cudaMemcpyHostToDevice);

    int* d_levels;
    cudaMalloc((void**)&d_levels, sizeof(int) * NUM_NODES);
    cudaMemcpy(d_levels, &levels, sizeof(int) * NUM_NODES, cudaMemcpyHostToDevice);

    int* d_currentLevel;
    cudaMalloc((void**)&d_currentLevel, sizeof(int));

    int* d_NUM_NODES;
    cudaMalloc((void**)&d_NUM_NODES, sizeof(int));
    cudaMemcpy(d_NUM_NODES, &NUM_NODES, sizeof(int), cudaMemcpyHostToDevice);


    int blocks = 1;
    int threads = NUM_NODES;

    int done;
    int* d_done;
    cudaMalloc((void**)&d_done, sizeof(int));

    int iterations = 0;

    do {
        iterations++;
        done = 1;

        cudaMemcpy(d_done, &done, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_currentLevel, &currentLevel, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_oldFrontier, oldFrontier, sizeof(int) * NUM_NODES, cudaMemcpyHostToDevice);

        for(i=0; i < NUM_NODES; i++){
                    printf("old frontier values [%d]: %d\n",i, oldFrontier[i]);
        }


        CUDA_BFS_KERNEL <<<blocks, threads >>>(d_vertices, d_edges, d_oldFrontier, d_newFrontier,
                                               d_visited, d_levels, d_currentLevel, d_done, d_NUM_NODES, d_num_edges);


        cudaMemcpy(&done, d_done , sizeof(int), cudaMemcpyDeviceToHost);

        cudaMemcpy(newFrontier, d_newFrontier, sizeof(int) * NUM_NODES, cudaMemcpyDeviceToHost);

        for(i=0; i < NUM_NODES; i++){
            printf("new frontier values [%d]: %d\n",i, newFrontier[i]);

            oldFrontier[i] = 0;
            if(newFrontier[i] == 1){
                oldFrontier[i] = 1;
                newFrontier[i] = 0;
            }

        }
        currentLevel++;
        } while (done == 0);

    cudaMemcpy(levels, d_levels, sizeof(int) * NUM_NODES, cudaMemcpyDeviceToHost);

    printf("Number of times the kernel is called : %d \n", iterations);

    printf("\nLevel:\n");
        for (int i = 0; i<NUM_NODES; i++)
            printf("node %d cost: %d\n", i, levels[i]);
        printf("\n");
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
        cudaMemcpy(d_newFrontier, &newFrontier, sizeof(int) * num_nodes, cudaMemcpyHostToDevice);

        cudaMalloc((void**)&d_levels, sizeof(int) * num_nodes);
        cudaMemcpy(d_levels, &levels, sizeof(int) * num_nodes, cudaMemcpyHostToDevice);

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

    int levels[num_nodes];
    int visited[num_nodes];
    int oldFrontier[num_nodes];
    int newFrontier[num_nodes];
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
     int subOldFrontier[elementsPerProc+1];
     int subNewFrontier[elementsPerProc+1];

     for(i=0; i < elementsPerProc; i++){
             subOldFrontier[i] = 0;
             subNewFrontier[i] = 0;
     }

    int blocks = 1;
    int threads = num_nodes;

    int done;
    int globalDone;

    int globalOldFrontier[num_nodes];
    int globalNewFrontier[num_nodes];

    int iterations = 0;
    printf("world_rank: %d\n", world_rank);

    do {
        if(iterations == 0 && world_rank == 0){
            iterations++;
            done = 1;

            cudaMemcpy(d_done, &done, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_currentLevel, &currentLevel, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_oldFrontier, oldFrontier, sizeof(int) * num_nodes, cudaMemcpyHostToDevice);

           for(i=0; i < num_nodes; i++){
               printf("old frontier values [%d]: %d\n",i, oldFrontier[i]);
            }

           CUDA_BFS_KERNEL <<<blocks, threads >>>(d_vertices, d_edges, d_oldFrontier, d_newFrontier,
                                            d_visited, d_levels, d_currentLevel, d_done, d_num_nodes, d_num_edges);

           cudaMemcpy(newFrontier, d_newFrontier, sizeof(int) * num_nodes, cudaMemcpyDeviceToHost);
           cudaMemcpy(&done, d_done , sizeof(int), cudaMemcpyDeviceToHost);

           for(i=0; i < num_nodes; i++){
               printf("new frontier values [%d]: %d\n",i, newFrontier[i]);

                   oldFrontier[i] = 0;
                   if(newFrontier[i] == 1){
                       oldFrontier[i] = 1;
                       newFrontier[i] = 0;
                   }
           }
           printf("\n");
           for(i=0; i < num_nodes; i++){
              printf("old frontier values [%d]: %d\n",i, oldFrontier[i]);
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
            printf("NEW KERNEL CALLED\n");
            printf("\n");
            CUDA_BFS_KERNEL <<<blocks, threads >>>(d_vertices, d_edges, d_oldFrontier, d_newFrontier,
                                                   d_visited, d_levels, d_currentLevel, d_done, d_num_nodes, d_num_edges);

            cudaMemcpy(newFrontier, d_newFrontier, sizeof(int) * num_nodes, cudaMemcpyDeviceToHost);
            cudaMemcpy(&done, d_done , sizeof(int), cudaMemcpyDeviceToHost);

            for(i=0; i < num_nodes; i++)
                printf("AFTER COPY Process %d : new frontier values [%d]: %d\n",world_rank, i, newFrontier[i]);

            MPI_Allreduce(newFrontier, globalNewFrontier, num_nodes, MPI_INT, MPI_LOR, MPI_COMM_WORLD);

            MPI_Allreduce(&done, &globalDone, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
            printf("PROCESS %d : Done: %d\n", world_rank, done);
            printf("GlobalDone: %d\n", globalDone);

            for(i=0; i < num_nodes; i++){
                printf("new frontier values [%d]: %d\n",i, newFrontier[i]);

                oldFrontier[i] = 0;
                if(newFrontier[i] == 1){
                    oldFrontier[i] = 1;
                    newFrontier[i] = 0;
                }
            }
            currentLevel++;
        }
    } while (globalDone == 0);

    cudaMemcpy(levels, d_levels, sizeof(int) * num_nodes, cudaMemcpyDeviceToHost);


    int globalLevels[num_nodes];
    //maybe reduce
    //MPI_Gather(&levels, num_nodes, MPI_INT, globalLevels, num_nodes, MPI_INT, 0,MPI_COMM_WORLD);

    if(world_rank == 0){
        printf("Number of times the kernel is called : %d \n", iterations);
    }

    if(world_rank == 0){
        printf("\nLevel:\n");
        for (int i = 0; i<num_nodes; i++)
            printf("Process: %d node %d cost: %d\n",world_rank, i, levels[i]);
        printf("\n");
    }
}