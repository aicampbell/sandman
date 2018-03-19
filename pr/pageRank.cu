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

__global__ void CUDA_ITERATE_KERNEL(int* d_vertices, int* d_destinations, int* d_y, int* d_out_degrees, int d_num_vertices){

    int d = 0.85;
    int i;

    //Set the thread id
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if(d_out_degrees[id] != 0){
        for(i = d_vertices[id]; i < d_vertices[id +1]; i++){
            d_y[i] += d_vertices[id] / d_out_degrees[id];
        }
    }
    d_y[id] = ((1 - d) / d_num_vertices) + (d * d_y[id]);
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

__global__ void CUDA_DEGREE_KERNEL(int* d_out_degrees, int* d_vertices, int* d_num_vertices, int* d_num_edges){
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if(id == *d_num_vertices){
        d_out_degrees[id] = *d_num_edges - d_vertices[id];
    }
    else{
        d_out_degrees[id] = d_vertices[id+1] - d_vertices[id];
    }
}


void getDegrees(int* outDegrees, int* vertices, int maxVertices, int maxEdges){

    cudaMalloc((void**)&d_out_degrees, sizeof(int) * maxVertices);
    cudaMemcpy(d_out_degrees, outDegrees, sizeof(int) * maxVertices, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_vertices, sizeof(int) * maxVertices);
    cudaMemcpy(d_vertices, vertices, sizeof(int) * maxVertices, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_num_edges, sizeof(int));
    cudaMemcpy(d_num_edges, &maxEdges, sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_num_vertices, sizeof(int));
    cudaMemcpy(d_num_vertices, &maxVertices, sizeof(int), cudaMemcpyHostToDevice);

    //Call the kernel
    CUDA_DEGREE_KERNEL <<<1, maxVertices >>>(d_out_degrees, d_vertices, d_num_vertices, d_num_edges);

    //Transfer the outDegrees Back
    cudaMemcpy(outDegrees, d_out_degrees, sizeof(int) * maxVertices, cudaMemcpyDeviceToHost);

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

void pageRank(int* vertices, int maxVertices, int* edges, int maxEdges){
    int* outDegrees = (int*)malloc(sizeof(int) * maxVertices);

    int maxIterations = 100;
    int iteration = 0;
    float tol = 0.0000005;
    float* y;
    float delta = 2;

    //getDegrees(outDegrees, vertices, maxVertices, maxEdges);

    //while(iteration < maxIterations && delta > tol){

    //    //call kernel
    //    cuda

    //    //constants (1-d)v[i] added in separately.
    //    float weight = 1.0f - sum(y, maxVertices); //ensure y[] sums to 1
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
