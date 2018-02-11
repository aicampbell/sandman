#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int controlData[800][2];
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

    int nodes[maxNodes];
    int edges[maxEdges];
    int edge = 0;
    int source = 1;

    for(i=0; i <= maxNodes;i++){
        nodes[i] = -1;
    }


    while ((fscanf(f, "%d %d", &controlData[i][0], &controlData[i][1])) != EOF) {
        i++;
    }

    int currentPos = 0;
    for(i=source; i <= maxNodes;i++){
        if(i == source) {
            nodes[i] = currentPos;
        }

        for(j = 0; j < maxEdges; j++) {
            if (i == controlData[j][0]) {
                edges[edge] = controlData[j][1];
                edge++;
                currentPos++;

            }


        for(j = 0; j < maxEdges ; j++) {
            if (i == controlData[j][0]) {
                edges[edge] = controlData[j][1];
                edge++;
                length++;
                if(i < maxNodes -1) {
                    nodes[i + 1] += 1;
                }
                else{
                    nodes[i] += 1;
                }
            }
        }
    }

    printf("\n");
    printf("\n");
    printf("Vertices:\n");
    for(i =0; i < maxNodes; i++){
        printf("%d  %d\n", i, nodes[i]);
    }

    printf("\n");
    printf("\n");
    printf("Edges:\n");
    for(i =0; i < maxEdges; i++){
        printf("%d\n", edges[i]);
    }


    fclose (f);
        return 0;
}