#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int controlData[800][2];
int maxNodes, maxEdges;

int readInputFile(){
    FILE *f;
    f = fopen("chesapeak-sorted.mtx", "r");
    if (f == NULL)
        return 0;

    int r;
    int i = 0;


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

int main() {


    int i;

    readInputFile();


    int nodes[maxNodes];
    int edges[maxEdges];
    int edge = 0;
    int source = controlData[0][0];

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
}