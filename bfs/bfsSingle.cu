#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "bfs.h"
#include <mpi.h>

__global__ void CUDA_BFS_KERNEL(int *d_vertices, int *d_edges, int* d_oldFrontier, int* d_newFrontier, int* d_visited,
     int* d_levels, int *d_currentLevel, int *d_done, int *d_NUM_NODES){
    int i;

    //Set the thread id
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id >= *d_NUM_NODES)
        *d_done = 0;

    d_newFrontier[id] = 0;

    if (d_oldFrontier[id] == 1 && d_visited[id] == 1){
        //printf("Node order: %d \n", id); //This printf gives the order of vertices in BFS

        d_visited[id] = 1;
        d_levels[id] = *d_currentLevel; //set the cost of the current node



        int start = d_vertices[id];
        int end;
        if(d_vertices[id] >= *d_NUM_NODES){
            end = start;
        }
        else{
         end = d_vertices[id + 1];
          for(i = start; i < end; i++){
                     int nid = d_edges[i];
                     d_visited[nid] = 1;
                     d_newFrontier[nid] = 1;

                     *d_done = 0;
                 }
        }
        __syncthreads();
        if(d_vertices[id] >= *d_NUM_NODES){
            *d_done = 1;
        }
    }
    else{
        *d_done = 1;
    }
    printf("GPU newFrontier %d\n", d_newFrontier[id]);
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

//Example 1
    vertices[0] = 0;
    vertices[1] = 2;
    vertices[2] = 3;
    vertices[3] = 4;
    vertices[4] = 5;

    edges[0] = 1;
    edges[1] = 2;
    edges[2] = 4;
    edges[3] = 3;
    edges[4] = 4;

//Example 2
//    vertices[0] = 0;
//    vertices[1] = 3;
//    vertices[2] = 4;
//    vertices[3] = 5;
//    vertices[4] = 5;
//    vertices[5] = 5;


//    edges[0] = 1;
//    edges[1] = 2;
//    edges[2] = 3;
//    edges[3] = 4;
//    edges[4] = 5;



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
                                               d_visited, d_levels, d_currentLevel, d_done, d_NUM_NODES);


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

void distributedBFS(int NUM_NODES, int world_rank, int world_size){

    int i;

    int vertices[NUM_NODES];
    int levels[NUM_NODES];
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

//Example 1
//    vertices[0] = 0;
//    vertices[1] = 2;
//    vertices[2] = 3;
//    vertices[3] = 4;
//    vertices[4] = 5;

//    edges[0] = 1;
//    edges[1] = 2;
//    edges[2] = 4;
//    edges[3] = 3;
//    edges[4] = 4;

//Example 2
    vertices[0] = 0;
    vertices[1] = 3;
    vertices[2] = 4;
    vertices[3] = 5;
    vertices[4] = 5;
   vertices[5] = 5;


    edges[0] = 1;
    edges[1] = 2;
    edges[2] = 3;
    edges[3] = 4;
    edges[4] = 5;



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

     int elementsPerProc = (NUM_NODES / world_size);
     printf("elements per proc: %d\n", elementsPerProc);
     int* d_elementsPerProc;
     cudaMalloc((void**)&d_elementsPerProc, sizeof(int));
     cudaMemcpy(d_elementsPerProc, &elementsPerProc, sizeof(int), cudaMemcpyHostToDevice);

     int subOldFrontier[elementsPerProc];
     int subNewFrontier[elementsPerProc];

     for(i=0; i < elementsPerProc; i++){
             subOldFrontier[i] = 0;
             subNewFrontier[i] = 0;
     }

    int blocks = 1;
    int threads = NUM_NODES;

    int done;
    int* d_done;
    cudaMalloc((void**)&d_done, sizeof(int));

    int globalDone;

    int iterations = 0;
    printf("world_rank: %d\n", world_rank);

    do {

        if(iterations == 0 && world_rank == 0){
            iterations++;
            done = 1;

            cudaMemcpy(d_done, &done, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_currentLevel, &currentLevel, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_oldFrontier, oldFrontier, sizeof(int) * NUM_NODES, cudaMemcpyHostToDevice);

            for(i=0; i < NUM_NODES; i++){
                printf("old frontier values [%d]: %d\n",i, oldFrontier[i]);
            }


            CUDA_BFS_KERNEL <<<blocks, threads >>>(d_vertices, d_edges, d_oldFrontier, d_newFrontier,
                                            d_visited, d_levels, d_currentLevel, d_done, d_NUM_NODES);

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
            printf("\n");
            for(i=0; i < NUM_NODES; i++){
                            printf("old frontier values [%d]: %d\n",i, oldFrontier[i]);
            }
            currentLevel++;
        }

        MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if(iterations > 0){
        MPI_Bcast(&currentLevel, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Scatter(oldFrontier, elementsPerProc, MPI_INT, subOldFrontier, elementsPerProc, MPI_INT, 0, MPI_COMM_WORLD);

        iterations++;
        done = true;

        cudaMemcpy(d_done, &done, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_currentLevel, &currentLevel, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_oldFrontier, subOldFrontier, sizeof(int) * elementsPerProc, cudaMemcpyHostToDevice);

        for(i=0; i < elementsPerProc; i++){
            printf("Process %d : old frontier values [%d]: %d\n",world_rank, i, subOldFrontier[i]);
        }



        CUDA_BFS_KERNEL <<<blocks, threads >>>(d_vertices, d_edges, d_oldFrontier, d_newFrontier,
                                        d_visited, d_levels, d_currentLevel, d_done, d_elementsPerProc);


        cudaMemcpy(&done, d_done , sizeof(int), cudaMemcpyDeviceToHost);

        cudaMemcpy(subNewFrontier, d_newFrontier, sizeof(int) * elementsPerProc, cudaMemcpyDeviceToHost);

        for(i=0; i < elementsPerProc; i++){
                    printf("Process %d : sub new frontier values [%d]: %d\n",world_rank, i, subNewFrontier[i]);
        }

        MPI_Gather(&subNewFrontier, elementsPerProc, MPI_INT, newFrontier, elementsPerProc, MPI_INT, 0, MPI_COMM_WORLD);


        MPI_Allreduce(&done, &globalDone, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
        //printf("PROCESS %d : Done: %d\n", world_rank, done);
        //printf("GlobalDone: %d\n", globalDone);

        if(world_rank == 0){
            for(i=0; i < NUM_NODES; i++){
                printf("new frontier values [%d]: %d\n",i, newFrontier[i]);

                oldFrontier[i] = 0;
                if(newFrontier[i] == 1){
                    oldFrontier[i] = 1;
                    newFrontier[i] = 0;
                }

            }
        }

        if(world_rank != 0){
            for(i=0; i < 1; i++){
            oldFrontier[i] = 0;
            }
        }

        currentLevel++;

        for(i=0; i < NUM_NODES; i++){
                    printf("Process %d : END old frontier values [%d]: %d\n",world_rank, i, oldFrontier[i]);
                }

        }
     } while (globalDone == 0);

    cudaMemcpy(levels, d_levels, sizeof(int) * elementsPerProc, cudaMemcpyDeviceToHost);


    int globalLevels[NUM_NODES];
    MPI_Gather(&levels, elementsPerProc, MPI_INT, globalLevels, elementsPerProc, MPI_INT, 0,MPI_COMM_WORLD);

    if(world_rank == 0){
    printf("Number of times the kernel is called : %d \n", iterations);
    }

    if(world_rank == 0){
    printf("\nLevel:\n");
        for (int i = 0; i<NUM_NODES; i++)
            printf("Process: %d node %d cost: %d\n",world_rank, i, globalLevels[i]);
        printf("\n");
    }
}



int main(){
//    singleBFS(5);

int world_rank;
int world_size;

MPI_Init(NULL, NULL);
MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
MPI_Comm_size(MPI_COMM_WORLD, &world_size);


    distributedBFS(6, world_rank, world_size);

     MPI_Finalize();
}