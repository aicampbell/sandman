#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "bfs/bfs.h"
#include <mpi.h>

int controlData[800][2];
int maxNodes, maxEdges, r, i;

int *nodes;
int *partitionSizes;
int *partitionEdges;
int *size;

void readInputFile(){
    FILE *f;
    f = fopen("chesapeak-sorted.mtx", "r");

    fscanf(f, "%d %d %d", &maxNodes, &r, &maxEdges);
    printf("%d %d %d\n", maxNodes, r, maxEdges);


    while ((fscanf(f, "%d %d", &controlData[i][0], &controlData[i][1])) != EOF) {
        i++;
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

    if(nodes[vertex] == -1){
        return 0;
    }
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

void partitionByDestination(int *vertices, int numPartitions, int source){
    int averageDeg = maxEdges / numPartitions;
    printf("averageDeg Per Partition: %d\n", averageDeg);

    partitionEdges = (int *)malloc(numPartitions * sizeof(int));
    size = (int *)malloc(numPartitions * sizeof(int));

    int p = 0;
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
}

int main() {


    int i;

    readInputFile();

    int world_rank;
    int world_size;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    nodes = (int *)malloc(maxNodes * sizeof(int));
    partitionSizes = (int *)malloc(world_size * sizeof(int));

    int edges[maxEdges];
    int edge = 0;
    int source = controlData[0][0];

    if(world_rank == 0){
        for (i = 0; i <= maxNodes; i++) {
            nodes[i] = -1;
        }

        convertToCSR(source, maxNodes, maxEdges, nodes, edges);

        printf("\n");
        printf("\n");
        printf("Vertices:\n");
        for(i =0; i <= maxNodes; i++){
            printf("%d  %d\n", i, nodes[i]);
        }

        printf("Num degrees: vertex 11: %d\n", getDegree(38));


        partitionByDestination(nodes, world_size, source);

        printf("Sizes\n");
        for(i =0; i < world_size; i++){
            printf("%d  %d\n", i, size[i]);
        }

        printf("\n");
        printf("partitionEdges\n");
        for(i =0; i < world_size; i++){
            printf("%d  %d\n", i, partitionEdges[i]);
        }
    }
}