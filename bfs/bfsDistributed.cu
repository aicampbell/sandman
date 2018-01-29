#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define NUM_NODES 5

typedef struct
{
    int id;
    int start;     // Index of first adjacent node in Ea
    int length;    // Number of adjacent nodes
    bool visited;
    int cost;
} Node;

//Shared counter for updating array
    __shared__ int arrayCounter;

__global__ void CUDA_BFS_KERNEL(Node *d_node, int *d_edges, int* d_frontier, int *d_frontierSize, int* d_cost, bool *done)
{
    int i;

    if (threadIdx.x == 0){

        printf("CUDA NODES TO RUN:\n");
        for(i=0; i < *d_frontierSize; i++){
            printf("Node %d\n", d_node[i].id);
        }
        printf("\n");
    }

    if (threadIdx.x == 0)
        arrayCounter = 0;
    __syncthreads();

    //Set the thread id
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id > NUM_NODES)
        *done = false;

    if (d_node[id].visited == false){
        printf("Node order: %d \n", d_node[id].id); //This printf gives the order of vertices in BFS

        d_node[id].visited = true;
        d_node[id].cost = *d_cost; //set the cost of the current node

        __syncthreads();


        int start = d_node[id].start;
        int end = start + d_node[id].length;

        for(i = start; i < end; i++){
            int nid = d_edges[i];

            d_frontier[arrayCounter] = nid;
            atomicAdd(&arrayCounter, 1);
            *done = false;
        }

        *d_frontierSize = arrayCounter;

         if (threadIdx.x == 0){
                printf("CUDA NODES IN FRONTIER:\n");
                for(i=0; i < *d_frontierSize; i++){
                    printf("Node %d\n", d_frontier[i]);
                }
         }
        printf("length of frontier %d\n", arrayCounter);
    }
}


int main(){
    Node node[NUM_NODES];
    int frontier[NUM_NODES];

    int edges[NUM_NODES];

    node[0].start = 0;
    node[0].length = 2;

    node[1].start = 2;
    node[1].length = 1;

    node[2].start = 3;
    node[2].length = 1;
    //node[2].length = 0;

    node[3].start = 4;
    node[3].length = 1;

    node[4].start = 5;
    node[4].length = 0;

    edges[0] = 1;
    edges[1] = 2;
    edges[2] = 4;
    edges[3] = 3;
    edges[4] = 4;

    int i;
    for(i=0; i < NUM_NODES; i++){
        node[i].id = i;
        node[i].visited = false;
        node[i].cost = 0;
    }

    //set source
    int source = 0;


    Node nodesToCheck[NUM_NODES];
    nodesToCheck[source] = node[source];


    //COPY to GPU
    int nodeSize = sizeof(Node);
    int numBytes = NUM_NODES * nodeSize;
    Node* d_node;
    cudaMalloc((void**)&d_node, numBytes);


    int* d_edges;
    cudaMalloc((void**)&d_edges, sizeof(Node)*NUM_NODES);
    cudaMemcpy(d_edges, edges, sizeof(Node)*NUM_NODES, cudaMemcpyHostToDevice);

    int* d_frontier;
    cudaMalloc((void**)&d_frontier, sizeof(int)*NUM_NODES);
    cudaMemcpy(d_frontier, &frontier, sizeof(int)*NUM_NODES, cudaMemcpyHostToDevice);

    int frontierSize = 1;
    int * d_frontierSize;
    cudaMalloc((void**)&d_frontierSize, sizeof(int));

    int cost = 0;
    int* d_cost = 0;
    cudaMalloc((void**)&d_cost, sizeof(int));

    int blocks = 1;
    int threads = NUM_NODES;

    bool done;
    bool* d_done;
    cudaMalloc((void**)&d_done, sizeof(bool));
    printf("\n\n");
    int iterations = 0;

    do {
        iterations++;
        done = true;

        cudaMemcpy(d_done, &done, sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemcpy(d_frontierSize, &frontierSize, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cost, &cost, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_node, nodesToCheck, sizeof(Node) * frontierSize, cudaMemcpyHostToDevice);

        CUDA_BFS_KERNEL <<<blocks, threads >>>(d_node, d_edges, d_frontier, d_frontierSize, d_cost, d_done);

        cudaMemcpy(&done, d_done , sizeof(bool), cudaMemcpyDeviceToHost);
        cudaMemcpy(&nodesToCheck, d_node , sizeof(Node) * frontierSize, cudaMemcpyDeviceToHost);

        cudaMemcpy(&frontierSize, d_frontierSize, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&frontier, d_frontier , sizeof(int) * frontierSize, cudaMemcpyDeviceToHost);


        for(i=0; i < frontierSize; i++){
            int idx = nodesToCheck[i].id;
            node[idx] = nodesToCheck[i];

            printf("Node: %d in frontier: %d\n",i,frontier[i]);
            nodesToCheck[i] = node[frontier[i]];
        }
        cost++;
        } while (!done);

    //cudaMemcpy(node, d_node, numBytes, cudaMemcpyDeviceToHost);

    printf("Number of times the kernel is called : %d \n", iterations);

    printf("\nCost\n: ");
        for (int i = 0; i<NUM_NODES; i++)
            printf( "node %d cost: %d\n", i, node[i].cost);
        printf("\n");

}