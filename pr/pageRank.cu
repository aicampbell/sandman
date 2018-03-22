#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

int* d_vertices;
int* d_edges;
int* d_out_degrees;

int* d_num_vertices;
int* d_num_edges;
int* d_elementsPerProc;
int* d_done;

int world_rank;
int world_size;
int blocks;
int threads;


__global__ void CUDA_ITERATE_KERNEL(int* d_vertices, int* d_destinations, int* d_y, int* d_out_degrees, int* d_num_vertices){

    int d = 0.85;
    int i;

    //Set the thread id
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if(d_out_degrees[id] != 0){
        for(i = d_vertices[id]; i < d_vertices[id +1]; i++){
            d_y[i] += d_vertices[id] / d_out_degrees[id];
        }
    }
    d_y[id] = ((1 - d) / *d_num_vertices) + (d * d_y[id]);
}

__global__ void CUDA_WEIGHTS_KERNEL(float* d_weights, int d_weight, int* d_vertices, int d_num_vertices){
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    d_weights[id] += d_weight * d_vertices[id];
}

__global__ void CUDA_SCALE_SWAP_KERNEL(float* d_x, float* d_y, int d_weight){
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    d_x[id] = d_y[id] * d_weight;
    d_y[id] = 0.f;
}

void setup(int* vertices, int* destinations, int* out_degrees, int num_vertices, int num_edges){
    cudaMalloc((void**)&d_vertices, sizeof(int) * num_vertices);
    cudaMemcpy(d_vertices, vertices, sizeof(int) * num_vertices, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_destinations, sizeof(int) * num_edges);
    cudaMemcpy(d_destinations, destinations, sizeof(int) * num_edges, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_out_degrees, sizeof(int) * num_vertices);
    cudaMemcpy(d_out_degrees, out_degrees, sizeof(int) * num_vertices, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_num_vertices, sizeof(int));
    cudaMemcpy(d_num_vertices, &num_vertices, sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_num_edges, sizeof(int));
    cudaMemcpy(d_num_edges, &num_edges, sizeof(int), cudaMemcpyHostToDevice);
}

void iterate( int* y ){


    cudaMalloc((void**)&d_y, sizeof(int) * num_vertices);
    cudaMemcpy(d_destinations, destinations, sizeof(int) * num_vertices, cudaMemcpyHostToDevice);



    CUDA_ITERATE_KERNEL <<<blocks, threads >>> (d_vertices, d_destinations, d_y, d_out_degrees, d_num_vertices);

}

float sum(float* array, int length){
    float sum = 0.f;
    float err = 0.f;
    int i;
    for(i = 0; i < length; i++){
        float tmp = sum;
        float y = array[i] + err;
        sum = tmp + y;
        err = tmp - sum;
        err += y;
    }
    return sum;
}

//Calculate manhatten distance between input and output
float normdiff(float* input, float* output, int length){
    float d = 0.f;
    float err = 0.f;
    int i;
    for(i = 0; i < length; i++){
        float tmp = d;
        float y = abs(output[i] - input[i]) + err;
        d = tmp + y;
        err = tmp - d;
        err += y;
    }
    return d;
}

void pageRank(int* vertices, int maxVertices, int* edges, int maxEdges, int* outDegrees){

    int blocks = 1;
    int threads = maxVertices;

    int maxIterations = 100;
    int iteration = 0;
    float tol = 0.0000005;
    float* y;
    float delta = 2;

    setup();

    while(iteration < maxIterations && delta > tol){

        //call iterations
        iterate();

    //    //constants (1-d)v[i] added in separately.
        float weight = 1.0f - sum(y, maxVertices); //ensure y[] sums to 1
    //    CUDA_WEIGHTS_KERNEL();

    //    delta = normdiff(x, y, n);
    //    iteration++;

    //    //rescale to unit length
        //
    //    CUDA_SCALE_SWAP_KERNEL(float* x, float* y, int w)
    //}

    if(delta > tol){
        printf("No convergence");
    }
}
