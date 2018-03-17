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
    int* d_out_degrees;


    int* d_num_vertices;
    int* d_num_edges;
    int* d_elementsPerProc;
    int* d_done;

    int world_rank;
    int world_size;

__global__ void CUDA_ITERATE_KERNEL(int* d_vertices, int* d_destinations, int* d_results, int* d_out_degrees, int d_num_vertices){

    int d = 0.85;
    int i;

    //Set the thread id
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if(d_out_degrees[id] != 0){
        for(i = d_vertices[id]; i < d_vertices[id +1]; i++){
            output[i] += d_vertices[id] / d_out_degrees[id];
        }
    }
    output[id] = ((1 - d) / d_num_vertices) + (d * output[id]);
}

__global__ void CUDA_WEIGHTS_KERNEL(float* d_weights, int d_weight, int* d_vertices, int d_num_vertices){
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    weights[id] += w * d_vertices[id];
}

__global__ void CUDA_SCALE_SWAP_KERNEL(float* x, float* y, int w){
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    x[id] = y[id] * w;
    y[id] = 0.f;
}

__global__ void CUDA_DEGREE_KERNEL(int* d_out_degrees, int* d_vertices, int d_num_vertices, int d_num_edges){
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if(id == d_num_vertices){
        d_out_degrees[id] = d_num_edges - d_vertices[id];
    }
    else{
        d_out_degrees[id] = d_vertices[id+1] - d_vertices[id];
    }
}


void getDegree(int* outDegrees, int* vertices, int maxVertices, int maxEdges){

    cudaMalloc((void**)&d_out_degrees, sizeof(int) * maxVertices);
    cudaMemcpy(d_out_degrees, outDegrees, sizeof(int) * maxVertices, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_vertices, sizeof(int) * maxVertices);
    cudaMemcpy(d_vertices, vertices, sizeof(int) * maxVertices, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_num_edges, sizeof(int));
    cudaMemcpy(d_num_edges, &maxEdges, sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_num_vertices, sizeof(int));
    cudaMemcpy(d_num_vertices, &maxVertices, sizeof(int), cudaMemcpyHostToDevice);

    //Call the kernel
    CUDA_DEGREE_KERNEL(d_out_degrees, d_vertices, d_num_vertices, d_num_edges);

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
        float y = abs(output[i] - a[i]) + err;
        d = tmp + y;
        err = tmp - d;
        err += y;
    }
    return d;
}

void pageRank(int* vertices, int, maxVertices, int* edges, int maxEdges){
    int* outDegrees = malloc(sizeof(int) * maxVertices);

    int maxIterations = 100;
    int iteration;
    int tol = 0.0000005;
    float* y =
    float delta;

    while(iteration < maxIterations && delta > tol){

        //call kernel
        cuda

        //constants (1-d)v[i] added in separately.
        float weight = 1.0f - sum(y, maxVertices); //ensure y[] sums to 1
        CUDA_WEIGHTS_KERNEL();

        delta = normdiff(x, y, n);
        iteration++;

        //rescale to unit length
        //
    }

    if(delta > tol){
        printf("No convergence");
    }
}
