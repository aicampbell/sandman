#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "bfs/bfsSingle.cu"
#include <mpi.h>

int controlData[800][2];
int maxNodes, maxEdges, r, i;

int *nodes;
int *partitionSizes;
int *partitionEdges;
int *size;
int *starts;

void readInputFile(){
    FILE *f;
    f = fopen("test-sm.mtx", "r");

    fscanf(f, "%d %d %d", &maxNodes, &r, &maxEdges);
    printf("%d %d %d\n", maxNodes, r, maxEdges);


    while ((fscanf(f, "%d %d", &controlData[i][0], &controlData[i][1])) != EOF) {
        i++;
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

void convertToCSR(int source, int maxNodes, int maxEdges, int nodes[], int edges[]) {
    int i;
    int j;
    int edge = 0;
    int currentPos = 0;
    for (i = source; i <= maxNodes; i++) {
        if (i == source) {
            nodes[i] = currentPos;
        }

        for (j = 0; j <= maxEdges; j++) {
            if (i == controlData[j][0]) {
               //Sets edges[0] to the first position
                edges[edge] = controlData[j][1];
                edge++;
                currentPos++;

                if (i < maxNodes) {
                    if (nodes[i + 1] < 0) {
                        nodes[i + 1] = 0;
                    }
                   nodes[i + 1] = currentPos;
                }
                    //Last node so just max edges - second last node value
                else {
                    if (nodes[i] < 0) {
                        nodes[i] = currentPos;
                    }
                }
            }
        }
    }
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

int main() {


    int i;

    readInputFile();

    int world_rank;
    int world_size;
    int* localEdges;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    nodes = (int *)malloc(maxNodes * sizeof(int));
    partitionSizes = (int *)malloc(world_size * sizeof(int));
    starts = (int *)malloc(world_size * sizeof(int));

    partitionEdges = (int *)malloc(world_size * sizeof(int));


    int edges[maxEdges];
    int edge = 0;
    int source = controlData[0][0];

    if(world_rank == 0){
        for (i = 0; i <= maxNodes; i++) {
            nodes[i] = 0;
        }

        convertToCSR(source, maxNodes, maxEdges, nodes, edges);

        printf("\n");

        partitionByDestination(nodes, world_size);
        //Sizes can be used to determine starting position for each process

    }

    //broadcast all sources, partitionEdges to all processes
    MPI_Bcast(nodes, maxNodes +1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(partitionEdges, world_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(starts, world_size, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    localEdges = (int *)malloc(200 * sizeof(int));


    //send edgesPerPartiton to the correct Process
    if(world_rank == 0){
       for(i = 1; i < world_size; i++){
            MPI_Send(edges + starts[i], partitionEdges[i], MPI_INT, i, 0, MPI_COMM_WORLD);
       }
       for(i =0; i < partitionEdges[0]; i++){
            localEdges[i] = edges[i];
       }
       printf("PRocess: %d, Partition Edges %d\n",world_rank, partitionEdges[world_rank] );
    }
    else{
        printf("PRocess: %d, Partition Edges %d\n",world_rank, partitionEdges[world_rank] );
        MPI_Recv(localEdges, partitionEdges[world_rank], MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }


    printf("Process %d: starting value %d\n", world_rank, localEdges[0]);

    //test(world_rank);
    distributedBFS(nodes, localEdges, 5, maxEdges, world_rank, world_size, 0);


    //Each process call kernel
    //End of kernel,  all to all?
    //For levels, need to take min of each value
}