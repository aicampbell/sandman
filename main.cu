#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "bfs/bfsSingle.cu"
#include <mpi.h>

int **graph;
int maxNodes, maxEdges, r, i;

int *nodes;
int *edges;
int *partitionSizes;
int *partitionEdges;
int *size;
int *starts;
int *verticesStarts;

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
        return maxEdges;
    }
    else{
        int max = starts[0];
        for(i = 1; i < numPartitions; i ++){
    	  if(starts[i] - starts[i-1] > max){
    	    max = starts[i] - starts[i-1];
    	  }
    	}
        assert( max <= maxEdges );
        return max;
    }
}

void computeStarts(int numPartitions, int* partitionEdges){
    int i;
    int startID = 0;
    for(i = 0; i < numPartitions; i++){

        starts[i] = startID;
        startID += partitionEdges[i];
        assert( startID <= maxEdges );
        printf("Start[%d] %d\n", i, starts[i]);
    }
}

void convertToCSR(int maxNodes, int maxEdges, int* vertices, int** graph) {
    int i;
    int j;
    int edge = 0;
    int stop = 0;

    for (i = 0; i <= maxNodes; i++) {

        vertices[i] = edge;
	
	// always skip to the place where we left off
        for (j = edge; j < maxEdges && stop == 0; j++) {

             if (i == graph[j][0]) {
                assert( graph[j][1] <= maxNodes );
               //Sets edges[0] to the first position
                edges[edge] = graph[j][1];
                assert( edges[edge] != -1);
                edge++;
             } else if(i < graph[j][0]){
                stop = 1; //next vertex has been found in graph, stop iterating through the loop
             }
        }
        stop = 0; //reset the flag to 0 after each vertex iteration
    }
    vertices[maxNodes] = maxEdges; //set the last vertex to the number of edges.
}

//Returns the number of outgoing edges for a vertex
int getDegree(int vertex){
    if(vertex < maxNodes){
        return nodes[vertex +1] - nodes[vertex];
    }
    else{
        return -1;
    }
}


void partitionByDestination(int *vertices, int numPartitions){
    int averageDeg = maxEdges / numPartitions; //Take average so we know approx. how many edges should be in each partition
    printf("averageDeg Per Partition: %d\n", averageDeg);

    size = (int *)malloc(numPartitions * sizeof(int));

    int p; //set the number of edges in each position to 0 by default.
    for(p=0; p < numPartitions; p++){
        partitionEdges[p] = 0;
    }

    int current = 0;
    size[0] = 0;

    int v;
    for(v = 0; v < maxNodes; v++){
        assert( current < numPartitions);
        partitionEdges[current] += getDegree(v); //get the number of outgoing edges for the vertex and add it to the current partition
        size[current] +=1; //number of vertices in this partition

	if(partitionEdges[current] >= averageDeg && current < numPartitions -1){
	//average number of edges has been met or exceeded for this partition. Start filling next partition
            current++;
            size[current] = 0;
        }
    }

    for(i = 0; i < numPartitions; i++){
        printf( "partition Edge %d = %d\n", i, partitionEdges[i] );
    }
    //Compute the starting positions (global edge offset) for each process
    computeStarts(numPartitions, partitionEdges);

    //Compute the global vertex offset for each partition.
    for(i = 0; i < numPartitions; i++){
        for(v=0; v < maxNodes; v++){
            if(starts[i] == vertices[v]){
                verticesStarts[i] = v;
                break;
            }
        }
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
    int num_rows = 63497070; //maximum number of rows that should be iterated through in a graph file. Should change this to something dynamic
    graph = (int**) malloc(sizeof(int*) * num_rows); //allocate memory for the graph
    for(i=0; i < num_rows; i++){
        graph[i] = (int*) malloc(sizeof(int) * 2 ); 
    }

    char* file = argv[1];
    readInputFile(file);

    nodes = (int *)malloc((maxNodes + 1) * sizeof(int)); //allocate vertices
    edges = (int *)malloc(maxEdges * sizeof(int)); //allocate edges
    partitionSizes = (int *)malloc(world_size * sizeof(int)); //allocate number of partitions
    starts = (int *)malloc(world_size * sizeof(int));  //allocate edge offsets
    verticesStarts = (int *)malloc(world_size * sizeof(int)); //allocate vertex offsets
    partitionEdges = (int *)malloc(world_size * sizeof(int)); //allocate actual partitions

    int edge = 0;
    int source = graph[0][0];

    //set all vertex values to 0 by default. Could likely use OpenMP to increase the efficiency here
    for (i = 0; i <= maxNodes; i++) {
        nodes[i] = 0;
    }

    convertToCSR(maxNodes, maxEdges, nodes, graph);
    printf("\n");

    printf("Partitioning by dest\n");
    partitionByDestination(nodes, world_size);

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
            assert( (starts[world_rank + 1] - starts[world_rank]) <= localEdgesSize );
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

    printf("Calling bfs gpu\n");

    int edgeOffset = starts[world_rank];
    //Need to pass edge offset
    distributedBFS(nodes, localEdges, maxNodes, maxEdges, verticesStarts, world_rank, world_size, source, edgeOffset);

    for(i=0; i < num_rows; i++){
            free(graph[i]);
    }
    free(graph);
    free(nodes);
    free(edges);
    free(localEdges);
    free(starts);

    MPI_Finalize();
}