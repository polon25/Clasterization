#include "cosine.h"

using namespace std;

float vectorLength(float* dataVector,int length){
	float sum=0;
	for (int i=0;i<length;i++){
		sum+=dataVector[i]*dataVector[i];
	}
	sum=sqrt(sum);
	return sum;
}

bool compareLength(dataVec data1, dataVec data2){
    return (data1.length < data2.length);
}
bool compareClusters(clusterInfo cluster1, clusterInfo cluster2){
    return (cluster1.size > cluster2.size);
}
bool compareObjects(objectInfo object1, objectInfo object2){
    return (object1.euclideanDistance < object2.euclideanDistance);
}

int countItem(float item, float *dataArray, int rows){
	int counter=0;
	for (int i=0; i<rows; i++){
		if (dataArray[i]==item)
			counter++;
	}
	return counter;
}
