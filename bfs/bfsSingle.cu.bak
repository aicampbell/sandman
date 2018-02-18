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
    bool frontier;
    int cost;
} Node;


__global__ void CUDA_BFS_KERNEL(Node *d_node, int *d_edges, bool *done)
{

    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id > NUM_NODES)
        *done = false;

    if (d_node[id].frontier == true && d_node[id].visited == false){
        printf("%d ", d_node[id].id); //This printf gives the order of vertices in BFS
        d_node[id].frontier = false;
        d_node[id].visited = true;
        __syncthreads();

        int i;
        int start = d_node[id].start;
        int end = start + d_node[id].length;

        for(int i = start; i < end; i++){
            int nid = d_edges[i];

            if (d_node[nid].visited == false){
                d_node[nid].cost = d_node[id].cost + 1;
                d_node[nid].frontier = true;
                *done = false;
             }
        }
    }
}


int main(){
    Node node[NUM_NODES];

        int edges[NUM_NODES];

        node[0].start = 0;
        node[0].length = 2;

        node[1].start = 2;
        node[1].length = 1;

        node[2].start = 3;
        node[2].length = 1;

        node[3].start = 4;
        node[3].length = 1;

        node[4].start = 5;
        node[4].length = 0;

        edges[0] = 1;
        edges[1] = 2;
        edges[2] = 4;
        edges[3] = 3;
        edges[4] = 4;

        int i=0;
        for(i=0; i < NUM_NODES; i++){
            node[i].id = i;
            node[i].frontier = false;
            node[i].visited = false;
            node[i].cost = 0;
        }

        //set source
        int source = 0;

        node[source].frontier = true;

        for(i=0; i < NUM_NODES; i++){
            printf("%d\n", node[i].id);
            printf("%d\n", node[i].frontier);
            printf("%d\n", node[i].visited);
            printf("%d\n", node[i].cost);
            printf("\n");
        }

     //COPY to GPU
     int nodeSize = sizeof(Node);
     int numBytes = NUM_NODES * nodeSize;
     Node* d_node;
     cudaMalloc((void**)&d_node, numBytes);
     cudaMemcpy(d_node, node, numBytes, cudaMemcpyHostToDevice);

    int* d_edges;
    cudaMalloc((void**)&d_edges, sizeof(Node)*NUM_NODES);
    cudaMemcpy(d_edges, edges, sizeof(Node)*NUM_NODES, cudaMemcpyHostToDevice);

    int blocks = 1;
    int threads = NUM_NODES;

    bool done;
    bool* d_done;
    cudaMalloc((void**)&d_done, sizeof(bool));
    printf("\n\n");
    int count = 0;

    do {
        count++;
        done = true;
        cudaMemcpy(d_done, &done, sizeof(bool), cudaMemcpyHostToDevice);
        CUDA_BFS_KERNEL <<<blocks, threads >>>(d_node, d_edges, d_done);
        cudaMemcpy(&done, d_done , sizeof(bool), cudaMemcpyDeviceToHost);

        } while (!done);
    cudaMemcpy(node, d_node, numBytes, cudaMemcpyDeviceToHost);

    printf("Number of times the kernel is called : %d \n", count);

    printf("\nCost: ");
        for (int i = 0; i<NUM_NODES; i++)
            printf( "%d    ", node[i].cost);
        printf("\n");

}