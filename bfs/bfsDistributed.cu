#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define NUM_NODES 6


__global__ void CUDA_BFS_KERNEL(int *d_vertices, int *d_edges, bool* d_oldFrontier, bool* d_newFrontier, bool* d_visited,
     int* d_levels, int *d_currentLevel, bool *d_done){
    int i;

    //Set the thread id
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= NUM_NODES)
        *d_done = false;

    d_newFrontier[id] = 0;

    if (d_oldFrontier[id] == true && d_visited[id] == true){
        printf("Node order: %d \n", id); //This printf gives the order of vertices in BFS

        d_visited[id] = true;
        d_levels[id] = *d_currentLevel; //set the cost of the current node

        printf("GPU LEvel: %d\n",d_levels[id] );


        int start = d_vertices[id];
        int end;
        if(d_vertices[id] >= NUM_NODES){
            end = start;
        }
        else{
         end = d_vertices[id + 1];
          for(i = start; i < end; i++){
                     int nid = d_edges[i];
                     d_visited[nid] = true;
                     d_newFrontier[nid] = true;

                     *d_done = false;
                 }
        }
        __syncthreads();
        if(d_vertices[id] >= NUM_NODES){
                    *d_done = true;
        }
    }
}


int main(){

    int i;

    int vertices[NUM_NODES];
    int levels[NUM_NODES];
    bool visited[NUM_NODES];
    bool oldFrontier[NUM_NODES];
    bool newFrontier[NUM_NODES];

    int currentLevel = 0;

    int edges[NUM_NODES];

    for(i=0; i < NUM_NODES; i++){
        levels[i] = -1;
        visited[i] = false;
        oldFrontier[i] = false;
        newFrontier[i] = false;
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

    oldFrontier[source] = true;
    visited[source] = true;

    //COPY to GPU
    int* d_vertices;
    cudaMalloc((void**)&d_vertices, sizeof(int) * NUM_NODES);
    cudaMemcpy(d_vertices, vertices, sizeof(int) * NUM_NODES, cudaMemcpyHostToDevice);

    int* d_edges;
    cudaMalloc((void**)&d_edges, sizeof(int) * NUM_NODES);
    cudaMemcpy(d_edges, edges, sizeof(int) * NUM_NODES, cudaMemcpyHostToDevice);

    bool* d_visited;
    cudaMalloc((void**)&d_visited, sizeof(bool) * NUM_NODES);
    cudaMemcpy(d_visited, visited, sizeof(bool) * NUM_NODES, cudaMemcpyHostToDevice);

    bool* d_oldFrontier; //memcpy in the iteration loop
    cudaMalloc((void**)&d_oldFrontier, sizeof(bool) * NUM_NODES);

    bool* d_newFrontier;
    cudaMalloc((void**)&d_newFrontier, sizeof(bool) * NUM_NODES);
    cudaMemcpy(d_newFrontier, &newFrontier, sizeof(bool) * NUM_NODES, cudaMemcpyHostToDevice);

    int* d_levels;
    cudaMalloc((void**)&d_levels, sizeof(int) * NUM_NODES);
    cudaMemcpy(d_levels, &levels, sizeof(int) * NUM_NODES, cudaMemcpyHostToDevice);

    int* d_currentLevel;
    cudaMalloc((void**)&d_currentLevel, sizeof(int));


    int blocks = 1;
    int threads = NUM_NODES;

    bool done;
    bool* d_done;
    cudaMalloc((void**)&d_done, sizeof(bool));

    int iterations = 0;

    do {
        iterations++;
        done = true;

        cudaMemcpy(d_done, &done, sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemcpy(d_currentLevel, &currentLevel, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_oldFrontier, oldFrontier, sizeof(bool) * NUM_NODES, cudaMemcpyHostToDevice);

        for(i=0; i < NUM_NODES; i++){
                    printf("old frontier values [%d]: %d\n",i, oldFrontier[i]);
        }


        CUDA_BFS_KERNEL <<<blocks, threads >>>(d_vertices, d_edges, d_oldFrontier, d_newFrontier,
                                               d_visited, d_levels, d_currentLevel, d_done);


        cudaMemcpy(&done, d_done , sizeof(bool), cudaMemcpyDeviceToHost);

        printf("done: %d\n", done);

        cudaMemcpy(newFrontier, d_newFrontier, sizeof(bool) * NUM_NODES, cudaMemcpyDeviceToHost);

        for(i=0; i < NUM_NODES; i++){
            printf("new frontier values [%d]: %d\n",i, newFrontier[i]);

            oldFrontier[i] = false;
            if(newFrontier[i] == true){
                oldFrontier[i] = true;
                newFrontier[i] = false;
            }

        }
        currentLevel++;
        } while (!done);

    cudaMemcpy(levels, d_levels, sizeof(int) * NUM_NODES, cudaMemcpyDeviceToHost);

    printf("Number of times the kernel is called : %d \n", iterations);

    printf("\nLevel:\n");
        for (int i = 0; i<NUM_NODES; i++)
            printf("node %d cost: %d\n", i, levels[i]);
        printf("\n");

}