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

            printf("s: %d\n", s);

            // new result += previous values / number of out degrees
            sum += d * d_x[s] / d_out_degrees[s];

             //printf("x[%d]: %.5f\n", s, d_x[s]);
             //printf("outDegree[%d]: %d\n", s, d_out_degrees[s]);
        }

        sum += (1 - d) / *d_num_vertices;

        d_y[id] = sum;
    }
}

__global__ void CUDA_WEIGHTS_KERNEL(double* d_y, double* d_weight, int* d_num_vertices){

    printf("Weight %1f\n", *d_weight);

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
    printf("SUM: %.5f\n", sum);
    printf("ERR: %.5f\n", err);
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
    int iteration = 0;
    double tol = 0.0000005;
    double* x = (double *) malloc(num_vertices * sizeof(double));
    double* y = (double *) malloc(num_vertices * sizeof(double));
    double delta = 2;

    setup(vertices, destinations, outDegrees, num_vertices, num_destinations, x);

    while(iteration < maxIterations && delta > tol){

        printf("Iteration: %d\n", iteration);

        //call iterations
        iterate(x, y, num_vertices);

        printf("x[0] = %.5f\n", x[0]);
        printf("x[1] = %.5f\n", x[1]);
        printf("x[2] = %.5f\n", x[2]);
        printf("x[3] = %.5f\n", x[3]);


        //printf("WEIGHTS\n");

        //constants (1-d)v[i] added in separately.
        double weight = 1.0f - sum(y, num_vertices); //ensure y[] sums to 1
        printf("Weight : %.5f\n", weight);

        cudaMemcpy(d_weight, &weight, sizeof(double), cudaMemcpyHostToDevice);
        CUDA_WEIGHTS_KERNEL<<<blocks, threads>>>(d_y, d_weight, d_num_vertices);
        cudaMemcpy(y, d_y, sizeof(double) * num_vertices, cudaMemcpyDeviceToHost);

        //printf("x[0] = %.5f\n", x[0]);
        //printf("x[1] = %.5f\n", x[1]);
        //printf("x[2] = %.5f\n", x[2]);
        //printf("x[3] = %.5f\n", x[3]);

        printf("y[0] = %.5f\n", y[0]);
        printf("y[1] = %.5f\n", y[1]);
        printf("y[2] = %.5f\n", y[2]);
        printf("y[3] = %.5f\n", y[3]);

        delta = normdiff(x, y, num_vertices);
        printf("Delta: %1f\n", delta);

        //rescale to unit length

        weight = 1.0f / sum(y, num_vertices);
        cudaMemcpy(d_weight, &weight, sizeof(double), cudaMemcpyHostToDevice);
        CUDA_SCALE_SWAP_KERNEL<<<blocks, threads>>>(d_x, d_y, d_weight);
        cudaMemcpy(x, d_x, sizeof(double) * num_vertices, cudaMemcpyDeviceToHost);
        cudaMemcpy(y, d_y, sizeof(double) * num_vertices, cudaMemcpyDeviceToHost);

        printf("After swap\n");

        printf("x[0] = %.5f\n", x[0]);
        printf("x[1] = %.5f\n", x[1]);
        printf("x[2] = %.5f\n", x[2]);
        printf("x[3] = %.5f\n", x[3]);

        printf("y[0] = %.5f\n", y[0]);
        printf("y[1] = %.5f\n", y[1]);
        printf("y[2] = %.5f\n", y[2]);
        printf("y[3] = %.5f\n", y[3]);

        iteration++;
    }

    if(delta > tol){
        printf("No convergence\n");
    }
    else{
        printf("Convergence Mofo\n");

        int i;
        for(i =0; i < num_vertices; i++){
            printf("x[%d] = %.5f\n", i, x[i]);
        }
    }
}
