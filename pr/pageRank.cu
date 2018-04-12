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
int* d_destinations;

//current pr values
double* d_x;
//new pr values
double* d_y;

double* d_weight;

int* d_num_vertices;
int* d_num_edges;
int* d_elementsPerProc;
int* d_done;

int world_rank;
int world_size;
int blocks;
int threads;

void log(double* x, int num_vertices){
    int i;
    for(i=0; i < num_vertices; i++){
        printf("x[%d] = %1f\n", i, x[i]);
    }
}

__global__ void CUDA_INIT_PR_VALUES(double* d_x, int* d_num_vertices){
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    d_x[id] = 1.0f / *d_num_vertices;

}

//d_y is the output page rank
//d_x is the old values
//d_destinations is the edges
//d_vertices is the sources
__global__ void CUDA_ITERATE_KERNEL(int* d_vertices, int* d_destinations, double* d_x, double* d_y, int* d_out_degrees, int* d_num_vertices){

    double d = 0.85;
    int i;
    int s;
    double sum = 0;

    //Set the thread id
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if(d_out_degrees[id] != 0){
        for(i = d_vertices[id]; i < d_vertices[id +1]; i++){
            s = d_destinations[i];

            if(d_out_degrees[s] != 0){
                // new result += previous values / number of out degrees
                sum += d * d_x[s] / d_out_degrees[s];
            }
        }
        //Likely need to add this outside of kernel when using MPI. Talk with Hans as this is constant
        sum += (1 - d) / *d_num_vertices;

        d_y[id] = sum;
    }
}

__global__ void CUDA_WEIGHTS_KERNEL(double* d_y, double* d_weight, int* d_num_vertices){

    int id = threadIdx.x + blockIdx.x * blockDim.x;

    d_y[id] += *d_weight * (1.0f / *d_num_vertices);

}

__global__ void CUDA_SCALE_SWAP_KERNEL(double* d_x, double* d_y, double* d_weight){
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    d_x[id] = *d_weight * d_y[id];
    d_y[id] = 0;
}

void setup(int* vertices, int* destinations, int* out_degrees, int num_vertices, int num_edges, double* x){
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

    cudaMalloc((void**)&d_x, sizeof(double) * num_vertices);
    cudaMemcpy(d_x, x, sizeof(double) * num_vertices, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_y, sizeof(double) * num_vertices);

    cudaMalloc((void**)&d_weight, sizeof(double));

    CUDA_INIT_PR_VALUES <<<blocks, threads>>> (d_x, d_num_vertices);
    cudaMemcpy(x, d_x, sizeof(double) * num_vertices, cudaMemcpyDeviceToHost);

}

void iterate(double* x, double* y, int num_vertices){

    cudaMemcpy(d_x, x, sizeof(double) * num_vertices, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, sizeof(double) * num_vertices, cudaMemcpyHostToDevice);

    CUDA_ITERATE_KERNEL <<<blocks, threads >>> (d_vertices, d_destinations, d_x, d_y, d_out_degrees, d_num_vertices);

    cudaMemcpy(x, d_x, sizeof(double) * num_vertices, cudaMemcpyDeviceToHost);
    cudaMemcpy(y, d_y, sizeof(double) * num_vertices, cudaMemcpyDeviceToHost);

}

float sum(double* array, int length){
    double sum = 0.f;
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
float normdiff(double* input, double* output, int length){
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



void pageRank(int* vertices, int num_vertices, int* destinations, int num_destinations, int* outDegrees){

    blocks = 1;
    threads = num_vertices;

    int maxIterations = 100;
    int iteration = 1;
    double tol = 0.0000005;
    double* x = (double *) malloc(num_vertices * sizeof(double));
    double* y = (double *) malloc(num_vertices * sizeof(double));
    double delta = 2;

    setup(vertices, destinations, outDegrees, num_vertices, num_destinations, x);

    while(iteration < maxIterations && delta > tol){

        log(y, num_vertices);
        //call iterations
        iterate(x, y, num_vertices);

        /**
        * Use MPI reduction here to do summation.
        * Then loop through and add  "sum += (1 - d) / *d_num_vertices;"
        *
        */

        //log(x, num_vertices);


        double weight = 1.0f - sum(y, num_vertices); //ensure y[] sums to 1

        cudaMemcpy(d_weight, &weight, sizeof(double), cudaMemcpyHostToDevice);
        CUDA_WEIGHTS_KERNEL<<<blocks, threads>>>(d_y, d_weight, d_num_vertices);
        cudaMemcpy(y, d_y, sizeof(double) * num_vertices, cudaMemcpyDeviceToHost);

        delta = normdiff(x, y, num_vertices);
        printf("Iteration: %d - Delta: %1f\n", iteration, delta);

        //rescale to unit length
        weight = 1.0f / sum(y, num_vertices);
        cudaMemcpy(d_weight, &weight, sizeof(double), cudaMemcpyHostToDevice);
        CUDA_SCALE_SWAP_KERNEL<<<blocks, threads>>>(d_x, d_y, d_weight);
        cudaMemcpy(x, d_x, sizeof(double) * num_vertices, cudaMemcpyDeviceToHost);
        cudaMemcpy(y, d_y, sizeof(double) * num_vertices, cudaMemcpyDeviceToHost);

        iteration++;
    }

    if(delta > tol){
        printf("\n");
        printf("No convergence\n");
    }
    else{
        printf("\n");
        printf("Convergence at iteration %d \n", iteration - 1);
        printf("\n");
        printf("Values:\n");

        int i;
        for(i =0; i < num_vertices; i++){
            printf("x[%d] = %.5f\n", i, x[i]);
        }
    }
}
