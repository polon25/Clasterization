#ifndef COSINE_H_
#define COSINE_H_

#include <iostream>
#include <math.h>

struct dataVec{//Data structure
	float* vector;
	float length;
};

struct clusterInfo{//Info about cluster
	int id;
	int size;
};

struct objectInfo{//Info about cluster
	int id;
	float cosineVal;
};

//Calculate length of vector from point 0
float vectorLength(float* dataVector,int length);
bool compareLength(dataVec data1, dataVec data2);
bool compareClusters(clusterInfo cluster1, clusterInfo cluster2);
bool compareObjects(objectInfo object1, objectInfo object2);
int countItem(float item, float *dataArray, int rows);

#endif /* COSINE_H_ */
