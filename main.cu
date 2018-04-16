#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
//#include "bfs/bfsSingle.cu"
#include <mpi.h>

int **graph;
int maxNodes, maxEdges, r, i;

int *nodes;
int *edges;
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
        assert( graph[i][0] <= maxNodes );
        assert( graph[i][1] <= maxNodes );
        i++;
    }
    printf("Graph loaded\n");
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

void convertToCSR(int maxNodes, int maxEdges, int* vertices, int** graph) {
    int i;
    int j;
    int edge = 0;
    //int* test = (int *)malloc(maxEdges * sizeof(int));

    for (i = 0; i < maxNodes; i++) {
        vertices[i] = edge;

        for (j = 0; j < maxEdges; j++) {
            if (i == graph[j][0]) {
               //Sets edges[0] to the first position
                assert( graph[j][1] <= maxNodes );
                edges[edge] = graph[j][1];
                //test[edge] = graph[j][1];
                edge++;
            }
        }
    }
    //printf("test[0] from method: %d\n", test[0]);

    printf("i: %d, j: %d, edge: %d - edges[0] from method: %d\n",i,j, edge, edges[5000]);
    printf("vertices[0] from method: %d\n", vertices[0]);
    printf("Graph[0][1] from method: %d\n", graph[0][1]);
//    printf("i: %d, j: %d, edge: %d - edges[0] from method: %d\n",i,j, edge, edges[0]);

    vertices[maxNodes] = maxEdges;
}

int getDegree(int vertex){

    if(vertex < maxNodes){
        return nodes[vertex +1] - nodes[vertex];
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
    for(v = 0; v < maxNodes + 1; v++){
        assert( current < numPartitions);
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
    printf("rank=%d size=%d\n",world_rank,world_size);

    int i;
    int  num_rows = 30000;
    graph = (int**) malloc(sizeof(int*) * num_rows);
    for(i=0; i < num_rows; i++){
        graph[i] = (int*) malloc(sizeof(int) * 2 );
    }


    char* file = argv[1];
    readInputFile(file);

    nodes = (int *)malloc((maxNodes + 1) * sizeof(int));
    edges = (int *)malloc(maxEdges * sizeof(int));
    partitionSizes = (int *)malloc(world_size * sizeof(int));
    starts = (int *)malloc(world_size * sizeof(int));

    partitionEdges = (int *)malloc(world_size * sizeof(int));

    int edge = 0;
    int source = graph[0][0];

    for (i = 0; i < maxNodes; i++) {
        nodes[i] = 0;
    }


    printf("CSR\n");
    convertToCSR(maxNodes, maxEdges, nodes, graph);
    printf("\n");

//    for(i = 0; i < 30; i++){
//        printf("vertex[%d]: %d\n", i, nodes[i]);
//    }
//
//    for(i = 0; i < 30; i++){
//            printf("edges[%d]: %d\n", i, edges[i]);
//        }

    printf("Partitioning by dest");
    partitionByDestination(nodes, world_size);

#if 1
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
#endif
    printf("Calling bfs gpu\n");
    //printf("");
    //distributedBFS(nodes, localEdges, maxNodes, maxEdges, world_rank, world_size, source);
}