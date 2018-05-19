#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "pr/pageRank.cu"
#include <mpi.h>

int **graph;
int maxNodes, maxEdges, r, i;

int *nodes;
int *edges;
int *outDegrees;
int *partitionSizes;
int *partitionEdges;
int *size;
int *starts;
int *verticesStarts;

/*
* Read graph inputfile
*/
void readInputFile(char* file){
    FILE *f;
    f = fopen(file, "r");

    fscanf(f, "%d %d %d", &maxNodes, &r, &maxEdges);
    while ((fscanf(f, "%d %d", &graph[i][0], &graph[i][1])) != EOF) {
        assert( graph[i][0] <= maxNodes );
        assert( graph[i][1] <= maxNodes );
        i++;
    }
}

//Return the value of the largest partition. Used to allocate memory later
int getMaxLocalEdgesSize(int numPartitions){
    if(numPartitions == 1){
        return maxEdges; //Return num of edges as there is only 1 partition
    }
    else{
        int max = starts[0];
        for(i =1; i < numPartitions; i ++){
    	  if(starts[i] - starts[i-1] > max){
    	    max = starts[i] - starts[i-1]; //iterate through checking each partition and checking the max size
    	  }
    	}
        assert( max <= maxEdges ); //Safety for debugging and make sure partitioning was correct
        return max;
    }
}

/*
*Compute starting points/offsets for each partition so edges can be copied from the global edge array.
*/
void computeStarts(int numPartitions, int* partitionEdges){
    int i;
    int startID = 0;
    for(i = 0; i < numPartitions; i++){

        starts[i] = startID;
        startID += partitionEdges[i];
        assert( startID <= maxEdges );
    }
}

void convertToCSC(int maxNodes, int maxEdges, int vertices[], int edges[]) {
    int i;
    int j;
    int edge = 0;
    int stop = 0;

    for (i = 0; i <= maxNodes; i++) {
        vertices[i] = edge; //start at the point where the previous vertex left off

        for(j = edge; j < maxEdges && stop == 0; j++) {

            if (i == graph[j][1]){
                assert(graph[j][0] <= maxNodes);
               //Sets edges[0] to the first position
                edges[edge] = graph[j][0];
                assert( edges[edge] != -1);
                edge++;
             } else if(i < graph[j][1]){  //stop when the column value is greater than the vertex value. Stops unnecessary iterations
                stop = 1;
             }
        }
        stop = 0; //reset stop after each vertex
    }
    vertices[maxNodes] = maxEdges;
}
/*
*Return number of outgoing edges for a vertex
*/
void getDegrees(){
    for(i=0; i < maxNodes; i++){
        if(i < maxNodes){
            outDegrees[i] = nodes[i + 1] - nodes[i];
        } else{
            outDegrees[i] = -1;
        }
    }
}

void partitionByEdges(int *vertices, int* outDegrees, int numPartitions){
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
    for(v = 0; v < maxNodes; v++){
        assert( current < numPartitions);
        partitionEdges[current] += outDegrees[v];
        size[current] +=1;
        if(partitionEdges[current] >= averageDeg && current < numPartitions -1){
            current++;
            size[current] = 0;
        }
    }
    computeStarts(numPartitions, partitionEdges);

    for(i = 0; i < numPartitions; i++){
        for(v=0; v < maxNodes; v++){
            if(starts[i] == vertices[v]){
                verticesStarts[i] = v;
                break;
            }
        }
        printf("vertex starting position parition[%d]: %d\n", i, verticesStarts[i]);
    }
}

int main(int argc, char **argv) {

    int world_rank;
    int world_size;
    int* localEdges;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int i;
    int num_rows = 97890601;
    graph = (int**) malloc(sizeof(int*) * num_rows);
    for(i=0; i < num_rows; i++){
        graph[i] = (int*) malloc(sizeof(int) * 2 );
    }

    char* file = argv[1];
    readInputFile(file);
    printf("Graph loaded\n");

    nodes = (int *)malloc((maxNodes + 1) * sizeof(int));
    outDegrees = (int *)malloc((maxNodes) * sizeof(int));
    edges = (int *)malloc(maxEdges * sizeof(int));
    partitionSizes = (int *)malloc(world_size * sizeof(int));
    starts = (int *)malloc(world_size * sizeof(int));
    verticesStarts = (int *)malloc(world_size * sizeof(int));

    partitionEdges = (int *)malloc(world_size * sizeof(int));

    int edge = 0;
    int source = graph[0][0];

    for (i = 0; i <= maxNodes; i++) {
        nodes[i] = 0;
    }

    convertToCSC(maxNodes, maxEdges, nodes, edges);

    //Calculate outward degrees for each vertex
    getDegrees();

    //Partition by edges
    partitionByEdges(nodes, outDegrees, world_size);

    MPI_Barrier(MPI_COMM_WORLD);

    int localEdgesSize = getMaxLocalEdgesSize(world_size);

    //Added 1000 is for safety to make sure enough memory is allocated.
    localEdges = (int *)malloc((localEdgesSize + 1000) * sizeof(int));
    int offset = 0;
    if(world_rank < world_size - 1){
       offset = starts[world_rank];
       for(i = 0; i < (starts[world_rank + 1] - starts[world_rank]); i++){
            assert( starts[world_rank + 1] < maxEdges );
            assert( (starts[world_rank + 1] - starts[world_rank]) <= localEdgesSize );
            localEdges[i] = edges[i + offset];
       }
    }
    else{
        offset = starts[world_rank];
        for(i = 0; i < (maxEdges - starts[world_rank]) ; i++){
           assert( (maxEdges - starts[world_rank]) <= localEdgesSize );
           localEdges[i] = edges[i + offset];
        }
    }

    printf("\n");
    printf("Partitioning finished, Calling Page Rank");
    printf("\n");
    printf("\n");
    pageRank(nodes, maxNodes, edges, maxEdges, outDegrees, verticesStarts, world_rank, world_size, maxEdges);

    MPI_Finalize();
}