#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <time.h>

    int nodesPerProc;

    int* d_vertices;
    int* d_edges;
    int* d_visited;
    int* d_oldFrontier; //memcpy in the iteration loop
    int* d_newFrontier;
    int* d_levels;
    int* d_currentLevel;
    int* d_num_nodes;
    int* d_num_edges;
    int* d_done;
    int* d_process_id;
    //Number of nodes in the old frontier for each process
    int* d_nodes_per_process;
    int* d_edgeOffset;
    int* d_nodeOffset;

    int world_rank;
    int world_size;

__global__ void CUDA_BFS_KERNEL(int *d_vertices, int *d_edges, int* d_oldFrontier, int* d_newFrontier, int* d_visited,
     int* d_levels, int *d_currentLevel, int *d_done, int *d_num_nodes, int *d_num_edges, int *d_process_id,
     int *d_nodes_per_process, int *d_nodeOffset, int *d_edgeOffset){

    int i;

    //Set the thread id
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = *d_nodeOffset;

    if(id > *d_num_nodes)
        *d_done = 1;

    d_newFrontier[id + offset] = 0;
    __syncthreads();

    if (d_oldFrontier[id] == 1 && d_visited[id + offset] == 1){
        //printf("Node order: %d \n", id + offset); //This printf gives the order of vertices in BFS
        d_levels[id + offset] = *d_currentLevel; //set the level of the current node

        int start = d_vertices[id + offset];
        int end = d_vertices[id + 1 + offset];

        //need to set starting vertices like pr in main.cu

        //printf("rank: %d : start: %d\n", *d_process_id, start);
        //printf("rank: %d : end: %d\n", *d_process_id, end);
        //printf("rank: %d : edgeOffset: %d\n", *d_process_id, *d_edgeOffset );

        for(i = start; i < end; i++){
            int nid = d_edges[i - *d_edgeOffset];
            //printf("GPU Nid: %d\n", nid);
            printf("");
            if(d_visited[nid] == false){
                d_visited[nid] = 1;
                d_newFrontier[nid] = 1;
                *d_done = 0;
            }
        }
    }
    __syncthreads();

}

void sendToGPU(int* vertices, int* edges, int* visited, int* newFrontier, int* levels, int num_nodes, int num_edges,
                int* verticesStarts, int world_size, int world_rank, int edgeOffset){

    //COPY to GPU

        cudaMalloc((void**)&d_vertices, sizeof(int) * num_nodes);
        cudaMemcpy(d_vertices, vertices, sizeof(int) * num_nodes, cudaMemcpyHostToDevice);

        cudaMalloc((void**)&d_edges, sizeof(int) * num_edges);
        cudaMemcpy(d_edges, edges, sizeof(int) * num_edges, cudaMemcpyHostToDevice);

        cudaMalloc((void**)&d_visited, sizeof(int) * num_nodes);
        cudaMemcpy(d_visited, visited, sizeof(int) * num_nodes, cudaMemcpyHostToDevice);

        cudaMalloc((void**)&d_oldFrontier, sizeof(int) * num_nodes);

        cudaMalloc((void**)&d_newFrontier, sizeof(int) * num_nodes);
        cudaMemcpy(d_newFrontier, newFrontier, sizeof(int) * num_nodes, cudaMemcpyHostToDevice);

        cudaMalloc((void**)&d_levels, sizeof(int) * num_nodes);
        cudaMemcpy(d_levels, levels, sizeof(int) * num_nodes, cudaMemcpyHostToDevice);

        cudaMalloc((void**)&d_currentLevel, sizeof(int));

        //If vertices are odd, place last vertex in highest rank
        if(world_rank == (world_size -1)){
            nodesPerProc = num_nodes - verticesStarts[world_rank];
        }else{
            nodesPerProc = verticesStarts[world_rank + 1] - verticesStarts[world_rank];
        }
	assert( nodesPerProc > 0 );
        printf("Process %d nodes per proc: %d\n", world_rank, nodesPerProc);

        cudaMalloc((void**)&d_nodes_per_process, sizeof(int));
        cudaMemcpy(d_nodes_per_process, &nodesPerProc, sizeof(int), cudaMemcpyHostToDevice);

        cudaMalloc((void**)&d_process_id, sizeof(int));
        cudaMemcpy(d_process_id, &world_rank, sizeof(int), cudaMemcpyHostToDevice);

        cudaMalloc((void**)&d_num_nodes, sizeof(int));
        cudaMemcpy(d_num_nodes, &num_nodes, sizeof(int), cudaMemcpyHostToDevice);

        cudaMalloc((void**)&d_num_edges, sizeof(int));
        cudaMemcpy(d_num_edges, &num_edges, sizeof(int), cudaMemcpyHostToDevice);

        cudaMalloc((void**)&d_nodeOffset, sizeof(int));

        cudaMalloc((void**)&d_edgeOffset, sizeof(int));
        cudaMemcpy(d_edgeOffset, &edgeOffset, sizeof(int), cudaMemcpyHostToDevice);

        cudaMalloc((void**)&d_done, sizeof(int));
}


int factor(int length){
    int res = 1;
    int i;
    for(i=1; i <= length && i <= 1024; i++){
        if(length % i == 0){
            res = i;
        }
    }
    return res;
}

void distributedBFS(int* vertices, int* edges, int num_nodes, int num_edges, int* verticesStarts, int world_rank, int world_size, int source, int edgeOffset){

    int* globalLevels = (int *)malloc(num_nodes * sizeof(int));
    int offset;
    int* levels = (int *)malloc(num_nodes * sizeof(int));
    int* visited = (int *)malloc(num_nodes * sizeof(int));
    int* oldFrontier = (int *)malloc(num_nodes * sizeof(int));
    int* newFrontier = (int *)malloc(num_nodes * sizeof(int));

    int* globalOldFrontier = (int *)malloc(num_nodes * sizeof(int));
    int* globalNewFrontier = (int *)malloc(num_nodes * sizeof(int));
    int* globalVisited = (int *)malloc(num_nodes * sizeof(int));
    int* globalNodesPerProcs;
    int* displs = NULL;

    int currentLevel = 0;

    double startExecution = MPI_Wtime();

    int i;
    for(i=0; i < num_nodes; i++){
        levels[i] = -1;
        visited[i] = 0;
        globalVisited[i] = 0;
        globalOldFrontier[i] = 0;
        globalNewFrontier[i] = 0;
    }

    //Add the source vertex to the frontier and set visited to true
    globalOldFrontier[source] = 1;
    visited[source] = 1;

    //sentToGPU
    sendToGPU(vertices, edges, visited, newFrontier, levels, num_nodes, num_edges, verticesStarts, world_size, world_rank, edgeOffset);
    printf("Process %d edge offset: %d\n", world_rank, edgeOffset);

    if(world_rank == 0){
        globalNodesPerProcs =  (int* ) malloc ( world_size * sizeof(int) );
    }

    MPI_Gather(&nodesPerProc, 1, MPI_INT, globalNodesPerProcs, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    displs =  (int* ) malloc ( world_size * sizeof(int) );
    if(world_rank == 0){
        displs[0] = 0;

        for(i=1; i < world_size; i++){
            displs[i] = displs[i-1] + globalNodesPerProcs[i-1];
            printf("displs[%d]: %d\n", i, displs[i]);
        }
    }
    MPI_Bcast(displs, world_size, MPI_INT, 0, MPI_COMM_WORLD);
    offset = displs[world_rank];
    printf("Process %d offset: %d\n", world_rank, offset);

    //do nodeDisplacement on GPU
    cudaMemcpy(d_nodeOffset, &offset, sizeof(int), cudaMemcpyHostToDevice);

    //Allocate sub arrays for MPI
    int* subOldFrontier = (int *)malloc(nodesPerProc * sizeof(int));
    int* subNewFrontier = (int *)malloc(num_nodes * sizeof(int));

    //Set default new and oldFrontier values to 0
    memset(subOldFrontier, 0, nodesPerProc * sizeof(int));
    memset(subNewFrontier, 0, num_nodes * sizeof(int));

    //Only for first iteration with Process 0
    int threads = factor(num_nodes);
    int blocks = num_nodes/threads;

    printf("threads: %d\n", threads);
    printf("blocks: %d\n", blocks);

    int done;
    int globalDone;

    double startIterationTime;
    double endIterationTime;

    double startCommunicationTime;
    double endCommunicationTime;

    double startCommunicationTime1;
    double endCommunicationTime1;

    clock_t start;
    clock_t diff;
    double msec;

    int iterations = 0;

    do {
        if(iterations == 0 && world_rank == 0){

            startIterationTime = MPI_Wtime();
            iterations++;

            cudaMemcpy(d_done, &done, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_currentLevel, &currentLevel, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_oldFrontier, globalOldFrontier, sizeof(int) * num_nodes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_visited, visited, sizeof(int) * num_nodes, cudaMemcpyHostToDevice);

            start = clock();
            CUDA_BFS_KERNEL <<<blocks, threads >>>(d_vertices, d_edges, d_oldFrontier, d_newFrontier,
                                           d_visited, d_levels, d_currentLevel, d_done, d_num_nodes, d_num_edges,
                                           d_process_id, d_nodes_per_process, d_nodeOffset, d_edgeOffset);
            diff = clock() - start;
            msec = diff * 1000 / CLOCKS_PER_SEC;

            cudaMemcpy(&done, d_done , sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(globalNewFrontier, d_newFrontier, sizeof(int) * num_nodes, cudaMemcpyDeviceToHost);
            cudaMemcpy(visited, d_visited, sizeof(int) * num_nodes, cudaMemcpyDeviceToHost);

            //Reset the values:
            for(i = 0; i < num_nodes; i++){
                globalOldFrontier[i] = 0;
                if(globalNewFrontier[i] == 1){
                    globalOldFrontier[i] = 1;
                    globalNewFrontier[i] = 0;
                }
            }

            currentLevel++;
            endIterationTime = MPI_Wtime();
            printf("Iteration: %d - Time taken: %.6f s\n", iterations, endIterationTime - startIterationTime);
            printf("Process: %d -> Time taken %.20f milliseconds\n", world_rank, msec);
            //printf("Process: %d -> Vertices processed per msec  %.5f\n", world_rank, (double)1/(msec%1000));

        }

        MPI_Barrier(MPI_COMM_WORLD);

        startCommunicationTime = MPI_Wtime();

        MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Scatterv(globalOldFrontier, globalNodesPerProcs, displs, MPI_INT, subOldFrontier, nodesPerProc, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Allreduce(visited, globalVisited, num_nodes, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        endCommunicationTime = MPI_Wtime();

        //Reset global oldFrontier values to 0
        memset(globalOldFrontier, 0, num_nodes * sizeof(int));

        if(iterations > 0){
            startIterationTime = MPI_Wtime();

            MPI_Bcast(&currentLevel, 1, MPI_INT, 0, MPI_COMM_WORLD);

            int count = 0;
            for(i = 0; i < nodesPerProc + 1; i++){
                if(subOldFrontier[i] == 1){
                count++;
                }
            }
            printf("First scatterv : Process %d: sub old frontier count: %d\n", world_rank, count);

            iterations++;
            done = 1;

            cudaMemcpy(d_done, &done, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_currentLevel, &currentLevel, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_oldFrontier, subOldFrontier, sizeof(int) * nodesPerProc, cudaMemcpyHostToDevice);
            cudaMemcpy(d_visited, globalVisited, sizeof(int) * num_nodes, cudaMemcpyHostToDevice);

            threads = factor(nodesPerProc);
            blocks = nodesPerProc / threads;
            //printf("threads: %d\n", threads);
            //printf("blocks: %d\n", blocks);

            start = clock();
            CUDA_BFS_KERNEL <<<blocks, threads >>>(d_vertices, d_edges, d_oldFrontier, d_newFrontier,
                            d_visited, d_levels, d_currentLevel, d_done, d_num_nodes, d_num_edges,
                            d_process_id, d_nodes_per_process, d_nodeOffset, d_edgeOffset);
            diff = clock() - start;
            msec = diff * 1000 / CLOCKS_PER_SEC;
            //printf("Process: %d -> Time taken %d seconds %.5f milliseconds\n", world_rank, msec/1000, (double)(msec%1000));
            //printf("Process: %d -> Vertices processed per msec  %.5f\n", world_rank, (double)count/(msec%1000));

            cudaMemcpy(&done, d_done , sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(subNewFrontier, d_newFrontier, sizeof(int) * num_nodes, cudaMemcpyDeviceToHost);
            cudaMemcpy(visited, d_visited, sizeof(int) * num_nodes, cudaMemcpyDeviceToHost);
            cudaMemcpy(levels, d_levels, sizeof(int) * num_nodes, cudaMemcpyDeviceToHost);

            MPI_Allreduce(&done, &globalDone, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

            count = 0;
            for(i=0; i < num_nodes; i++){
              if(subNewFrontier[i] == 1 && levels[i] >-1){
                 count++;
                 //printf("Process %d: new frontier val: %d\n", world_rank, i);
              }
            }

            MPI_Barrier(MPI_COMM_WORLD);
            printf("Iteration %d : Process %d: new frontier count: %d\n", iterations, world_rank, count);

            startCommunicationTime1 = MPI_Wtime();

            MPI_Reduce(subNewFrontier, globalOldFrontier, num_nodes, MPI_INT, MPI_LOR, 0, MPI_COMM_WORLD);
            //Reset new subFrontier values to 0
            memset(subNewFrontier, 0, num_nodes * sizeof(int));


            MPI_Allreduce(visited, globalVisited, num_nodes, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

            endCommunicationTime1 = MPI_Wtime();

            currentLevel++;

        endIterationTime = MPI_Wtime();

        if(world_rank == 0){
            printf("Iteration: %d - Time taken: %.6f s\n", iterations, endIterationTime - startIterationTime);

            printf("Iteration: %d - Communication Time taken for scattering: %.6f s\n", iterations, endCommunicationTime - startCommunicationTime);
            printf("Iteration: %d - Communication Time taken for reducing: %.6f s\n", iterations, endCommunicationTime1 - startCommunicationTime1);

        }

     }

    } while (globalDone == 0);
    double endExecution = MPI_Wtime();

    if(world_rank == 0){
        printf("Total execution time : %.6f s \n", endExecution - startExecution);
        printf("Number of times the kernel is called : %d \n", iterations);

    }


    MPI_Reduce(levels, globalLevels, num_nodes, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    if(world_rank == 0){
        for(i=0; i < num_nodes; i++){

        if(globalLevels[i] == 0 && i != source){
            globalLevels[i] = -1;
        }
    }
    }

    if(world_rank == 0){
       printf("num_nodes: %d\n", num_nodes);
       printf("\nLevel:\n");
       for (i = 0; i < num_nodes; i++)
           printf("node %d level: %d\n", i, globalLevels[i]);
       printf("\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
}