#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "bfs/bfsSingle.cu"
#include <mpi.h>

int graph[800][2];
int maxNodes, maxEdges, r, i;

int *nodes;
int *partitionSizes;
int *partitionEdges;
int *size;
int *starts;

void readInputFile(char* file){
    FILE *f;
    f = fopen(file, "r");

    fscanf(f, "%d %d %d", &maxNodes, &r, &maxEdges);
    printf("%d %d %d\n", maxNodes, r, maxEdges);


    while ((fscanf(f, "%d %d", &graph[i][0], &graph[i][1])) != EOF) {
        i++;
    }
}

int getMaxLocalEdgesSize(int numPartitions){
    if(numPartitions == 1){
        printf("max size: %d\n", maxEdges);
        return maxEdges;
    }
    else{
        int max = starts[0];
        for(i=1; i < numPartitions; i++){
            if(i == numPartitions -1){
                if (maxEdges - starts[i] > max){
                    max = maxEdges -starts[i];
                }
            }
            else if (starts[i] - starts[i-1] > max){
                max = starts[i] - starts[i-1];
            }
        }
        printf("max size: %d\n", max);
        return max;
    }
}

void computeStarts(int numPartitions, int* partitionEdges){
    int i;
    int startID = 0;
    for(i = 0; i < numPartitions; i++){
        starts[i] = startID;
        startID += partitionEdges[i+1];
    }
}

void convertToCSR(int source, int maxNodes, int maxEdges, int vertices[], int edges[]) {
    int i;
    int j;
    int edge = 0;

    for (i = 0; i < maxNodes; i++) {
        vertices[i] = edge;

        for (j = 0; j <= maxEdges; j++) {
            if (i == graph[j][0]) {
               //Sets edges[0] to the first position
                edges[edge] = graph[j][1];
                edge++;
             }
        }
    }
    vertices[maxNodes] = maxEdges;
}

int getDegree(int vertex){

    if(vertex < maxNodes){
        return nodes[vertex +1] - nodes[vertex];
    }
    else if(vertex == maxNodes){
        return maxEdges - nodes[vertex];
    }
    else{
        return -1;
    }
}

void partitionByDestination(int *vertices, int numPartitions){
    int averageDeg = maxEdges / numPartitions;
    printf("averageDeg Per Partition: %d\n", averageDeg);

    size = (int *)malloc(numPartitions * sizeof(int));

    int p;
    for(p=0; p < numPartitions; p++){
        partitionEdges[p] = 0;
    }

    int current = 0;
    size[0] = 0;

    int v;
    for(v = 0; v <= maxNodes; v++){
        partitionEdges[current] += getDegree(v);
        size[current] +=1;
        if(partitionEdges[current] >= averageDeg && current < numPartitions -1){
            current++;
            size[current] = 0;
        }
    }
    computeStarts(numPartitions, partitionEdges);
}

int main(int argc, char **argv) {

    int world_rank;
    int world_size;
    int* localEdges;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int i;
    char* file = argv[1];
    readInputFile(file);

    nodes = (int *)malloc(maxNodes * sizeof(int));
    partitionSizes = (int *)malloc(world_size * sizeof(int));
    starts = (int *)malloc(world_size * sizeof(int));

    partitionEdges = (int *)malloc(world_size * sizeof(int));

    int edges[maxEdges];
    int edge = 0;
    int source = graph[0][0];

    for (i = 0; i < maxNodes; i++) {
        nodes[i] = 0;
    }

    convertToCSR(source, maxNodes, maxEdges, nodes, edges);
    printf("\n");

    partitionByDestination(nodes, world_size);
    //Sizes can be used to determine starting position for each process

    if(world_rank == 1){
        for (i = 0; i < world_size; i++) {
            printf("start [%d] : %d\n", i, starts[i]);
        }
    }

    //broadcast all sources, partitionEdges to all processes
    //MPI_Bcast(nodes, maxNodes +1, MPI_INT, 0, MPI_COMM_WORLD);
    //MPI_Bcast(partitionEdges, world_size, MPI_INT, 0, MPI_COMM_WORLD);
    //MPI_Bcast(starts, world_size, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    int localEdgesSize = getMaxLocalEdgesSize(world_size);
    printf("local size: %d\n", localEdgesSize);

    //Added 10 is for safety to make sure enough memory is allocated.
    localEdges = (int *)malloc((localEdgesSize + 10) * sizeof(int));

    if(world_rank <= world_size -2){
       for(i = starts[world_rank]; i < maxEdges ; i++){
            localEdges[i] = edges[i];
       }
    }
    else{
        for(i = starts[world_rank]; i <= maxEdges; i++){
            localEdges[i] = edges[i];
        }
    }

    if(world_rank == 1){
        printf("LocalEdges\n");
        for (i = starts[world_rank]; i < maxEdges; i++) {
            printf("%d\n", localEdges[i]);
        }
    }



    printf("Process %d: starting value %d\n", world_rank, localEdges[0]);
    distributedBFS(nodes, localEdges, maxNodes, maxEdges, world_rank, world_size, 0);


    //Each process call kernel
    //End of kernel,  all to all?
    //For levels, need to take min of each value
}