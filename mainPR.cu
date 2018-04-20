#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "pr/pageRank.cu"
#include <mpi.h>

int graph[800][2];
int maxNodes, maxEdges, r, i;

int *nodes;
int *edges;
int *outDegrees;
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

void convertToCSR(int maxNodes, int maxEdges, int vertices[], int edges[]) {
    int i;
    int j;
    int edge = 0;

    for (i = 0; i < maxNodes; i++) {
        vertices[i] = edge;

        for (j = 0; j < maxEdges; j++) {
            if (i == graph[j][1]) {
               //Sets edges[0] to the first position
                edges[edge] = graph[j][0];
                edge++;
             }
        }
    }
    vertices[maxNodes] = maxEdges;
}

void getDegrees(){

    for(i=0; i <= maxNodes; i++){
        if(i < maxNodes){
            outDegrees[i] = nodes[i +1] - nodes[i];
        }
        else if(i == maxNodes){
            outDegrees[i] = maxEdges - nodes[i];
        }
        else{
            outDegrees[i] = -1;
        }
    }
}

void partitionByDestination(int *vertices, int* outDegrees, int numPartitions){
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
        partitionEdges[current] += outDegrees[v];
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
    outDegrees = (int *)malloc(maxNodes * sizeof(int));
    edges = (int *)malloc(maxEdges * sizeof(int));
    partitionSizes = (int *)malloc(world_size * sizeof(int));
    starts = (int *)malloc(world_size * sizeof(int));

    partitionEdges = (int *)malloc(world_size * sizeof(int));

    int edge = 0;
    int source = graph[0][0];

    for (i = 0; i < maxNodes; i++) {
        nodes[i] = 0;
    }

    convertToCSR(maxNodes, maxEdges, nodes, edges);
    printf("\n");

    //Calculate outward degrees for each vertex
    getDegrees();

    //Partition
    partitionByDestination(nodes, outDegrees, world_size);

    MPI_Barrier(MPI_COMM_WORLD);

    int localEdgesSize = getMaxLocalEdgesSize(world_size);
    printf("local size: %d\n", localEdgesSize);

    //Added 10 is for safety to make sure enough memory is allocated.
    localEdges = (int *)malloc((localEdgesSize + 1000) * sizeof(int));
    int offset = 0;
    if(world_rank < world_size - 1){
       offset = starts[world_rank];
       for(i = 0; i < (starts[world_rank + 1] - starts[world_rank]); i++){
            assert( starts[world_rank + 1] < maxEdges );
            assert( (starts[world_rank + 1] - starts[world_rank]) < localEdgesSize );
            localEdges[i] = edges[i + offset];
       }
    }
    else{
        printf("starts: %d\n", starts[world_rank]);

        offset = starts[world_rank];
        for(i = 0; i < (maxEdges - starts[world_rank]) ; i++){
           assert( (maxEdges - starts[world_rank]) <= localEdgesSize );
           localEdges[i] = edges[i + offset];
        }
    }

    for(i=0; i < maxNodes; i++){
        printf("Degrees: %d\n", outDegrees[i]);
    }

    pageRank(nodes, maxNodes, edges, maxEdges, outDegrees);
}