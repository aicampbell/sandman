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
int* d_offset;

int world_rank;
int world_size;
int blocks;
int threads;
int offset;

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
__global__ void CUDA_ITERATE_KERNEL(int* d_vertices, int* d_destinations, double* d_x, double* d_y, int* d_out_degrees, int* d_num_vertices, int* d_offset){

    double d = 0.85;
    int i;
    int s;
    double sum = 0;

    //Set the thread id
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if(d_out_degrees[id + *d_offset ] != 0){
        for(i = d_vertices[id + *d_offset]; i < d_vertices[id +1 + *d_offset]; i++){
            s = d_destinations[i];

            if(d_out_degrees[s] != 0){
                // new result += previous values / number of out degrees
                sum += d * d_x[s] / d_out_degrees[s]; // Check this
            }
        }
        //Likely need to add this outside of kernel when using MPI. Talk with Hans as this is constant
        sum += (1 - d) / *d_num_vertices;

        d_y[id  + *d_offset] = sum;
    }
}
//Do ALL to all after this

__global__ void CUDA_WEIGHTS_KERNEL(double* d_y, double* d_weight, int* d_num_vertices){

    int id = threadIdx.x + blockIdx.x * blockDim.x;

    d_y[id] += *d_weight * (1.0f / *d_num_vertices);

}

__global__ void CUDA_SCALE_SWAP_KERNEL(double* d_x, double* d_y, double* d_weight){
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    d_x[id] = *d_weight * d_y[id];
    d_y[id] = 0;
}

void setup(int* vertices, int* destinations, int* out_degrees, int num_vertices, int num_edges, double* x, double* y, int offset,
            int numLocalVertices){
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

    cudaMalloc((void**)&d_offset, sizeof(int));
    cudaMemcpy(d_offset, &offset, sizeof(int), cudaMemcpyHostToDevice);

    CUDA_INIT_PR_VALUES <<<blocks, threads>>> (d_x, d_num_vertices);
    printf("Init PR values Complete\n");
    cudaMemcpy(x, d_x, sizeof(double) * num_vertices, cudaMemcpyDeviceToHost);
    cudaMemcpy(y, d_x, sizeof(double) * num_vertices, cudaMemcpyDeviceToHost);

    printf("Setup Complete\n");
}

void iterate(double* inX, double* outX, double* inY, double* outY, int num_vertices){

    cudaMemcpy(d_x, inX, sizeof(double) * num_vertices, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, inY, sizeof(double) * num_vertices, cudaMemcpyHostToDevice);

    CUDA_ITERATE_KERNEL <<<blocks, threads >>> (d_vertices, d_destinations, d_x, d_y, d_out_degrees, d_num_vertices, d_offset);

    cudaMemcpy(outX, d_x, sizeof(double) * num_vertices, cudaMemcpyDeviceToHost);
    cudaMemcpy(outY, d_y, sizeof(double) * num_vertices, cudaMemcpyDeviceToHost);

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



void pageRank(int* vertices, int num_vertices, int* destinations, int num_destinations, int* outDegrees, int* verticesStarts,
                int world_rank, int world_size){

    int numLocalVertices;
    int i;

    int maxIterations = 100;
    int iteration = 1;
    double tol = 0.0000005;
    double delta = 2;

    int localDispl = verticesStarts[world_rank];

    double* x = (double *) malloc( num_vertices * sizeof(double));
    double* y = (double *) malloc( num_vertices * sizeof(double));


    if(world_rank < world_size - 1){
        numLocalVertices = (verticesStarts[world_rank + 1] - verticesStarts[world_rank]);
    }
    else{
        numLocalVertices = (num_vertices - verticesStarts[world_rank]);
    }

    int *recvcounts = NULL;
    if(world_rank == 0){
        recvcounts = (int *) malloc(world_size * sizeof(int));
    }

    MPI_Gather(&numLocalVertices, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    if(world_rank == 0){
        for(i = 0; i < world_size; i++){
            printf("recv count [%d] = %d\n", i, recvcounts[i]);
        }
    }

    blocks =  numLocalVertices / 2;
    threads = 2;

    int totlen = 0;
    int* displs = NULL;
    double* globalY = NULL;
    double* globalX = NULL;

    globalX = (double* ) malloc ( num_vertices * sizeof(double) );
    globalY = (double* ) malloc ( num_vertices * sizeof(double) );

    if(world_rank == 0){
        displs = (int* ) malloc ( world_size * sizeof(int) );
        displs[0] = 0;
        totlen = recvcounts[0];


        for(i=1 ; i < world_size; i++){
            totlen += recvcounts[i];
            displs[i] = displs[i-1] + recvcounts[i-1];
        }
    }

    setup(vertices, destinations, outDegrees, num_vertices, num_destinations, x, y, localDispl, numLocalVertices);


    MPI_Gatherv(x, numLocalVertices, MPI_DOUBLE, globalX, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(y, numLocalVertices, MPI_DOUBLE, globalY, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(globalX, num_vertices, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(globalY, num_vertices, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    while(iteration < maxIterations && delta > tol){


        //call iterations
        iterate(globalX, x, globalY, y, num_vertices);
        MPI_Reduce(y, globalY, num_vertices, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if(world_rank == 0 && iteration > 1){
            for(i=0; i < num_vertices; i++){
                printf("globalY[%d]: %1f\n", i, globalY[i]);
        }
        }

        if(world_rank == 0){
            double weight = 1.0f - sum(globalY, num_vertices); //ensure y[] sums to 1
            printf("weight: %1f\n", weight);


        cudaMemcpy(d_weight, &weight, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, globalY, sizeof(double) * num_vertices, cudaMemcpyHostToDevice);

        blocks = num_vertices / 2;
        CUDA_WEIGHTS_KERNEL<<<blocks, threads>>>(d_y, d_weight, d_num_vertices);

        cudaMemcpy(globalY, d_y, sizeof(double) * num_vertices, cudaMemcpyDeviceToHost);

        delta = normdiff(globalX, globalY, num_vertices);
        printf("Iteration: %d - Delta: %1f\n", iteration, delta);

        //rescale to unit length
        weight = 1.0f / sum(globalY, num_vertices);
        printf("weight: %1f\n", weight);

        cudaMemcpy(d_weight, &weight, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, globalY, sizeof(double) * num_vertices, cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, globalX, sizeof(double) * num_vertices, cudaMemcpyHostToDevice);
        CUDA_SCALE_SWAP_KERNEL<<<blocks, threads>>>(d_x, d_y, d_weight);
        cudaMemcpy(globalX, d_x, sizeof(double) * num_vertices, cudaMemcpyDeviceToHost);
        cudaMemcpy(y, d_y, sizeof(double) * num_vertices, cudaMemcpyDeviceToHost);
        cudaMemcpy(globalY, d_y, sizeof(double) * num_vertices, cudaMemcpyDeviceToHost);
        }

        MPI_Bcast(&delta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(globalX, num_vertices, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(globalY, num_vertices, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(y, num_vertices, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        iteration++;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if(world_rank == 0){

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
            printf("x[%d] = %.5f\n", i, globalX[i]);
        }
    }
    }
}
