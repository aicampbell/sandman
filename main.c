#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int controlData[800][2];

typedef struct
{
    int val;
    int start;     // Index of first adjacent node in Ea
    int length;    // Number of adjacent nodes
} Node;

int main()
{

    FILE *f;

    f = fopen ("dataset/chesapeake.mtx", "r");
    if (f == NULL)
        return 0;

    int r;
    int i = 0;
    int j;

    int maxNodes, maxEdges;
    fscanf(f, "%d %d %d", &maxNodes, &r, &maxEdges);
    printf("%d %d %d\n", maxNodes, r, maxEdges);

    Node nodes[maxNodes +1];
    int edges[maxEdges];
    int edge = 0;
    int start = 1;


    while ((fscanf(f, "%d %d", &controlData[i][0], &controlData[i][1])) != EOF) {
        i++;
    }

    for(i=start; i <= maxNodes;i++){
        nodes[i].val = i;
        nodes[i].length = 0;
        for(j = 0; j < maxEdges ; j++) {
            if (i == controlData[j][0]) {
                edges[edge] = controlData[j][1];
                edge++;
                nodes[i].length+=1;
            }
        }
        if(i != start){
            nodes[i].start = nodes[i -1].start + nodes[i-1].length;
        }
        else{
            nodes[i].start = 0;
        }
        printf("Node[%d] start: %d\n", i, nodes[i].start);
        printf("Node[%d] length: %d\n", i, nodes[i].length);

    }

    printf("\n");
    for(i =0; i < maxEdges; i++){
        printf("%d\n", edges[i]);
    }


    fclose (f);
        return 0;
}