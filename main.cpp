//============================================================================
// Name        : main.cpp
// Author      : Jacek Pi³ka
// Description : DAMI Project
// Arguments: filePath, number of header rows, number of columns with labels, delimiter, epsilon, number of iteration since cluster number doesn't change
//============================================================================

#include "data.h"
#include "cosine.h"
#include <algorithm>

using namespace std;

int main (int argc, char *arg[]){
	//Arguments
	string filePath=arg[1];
	int headerRows=atoi(arg[2]); //Number of header lines
	int labelColumns=atoi(arg[3]); //Number of columns with labels
	char delimiter=*arg[4]; //Delimiter char
	float epsilon=atof(arg[5]); //Epsilon value
	int tWait=atof(arg[6]); //Number of iteration to wait since stopping changing clusters number to stop algorithm

	/**
	 * Importing data from file
	 */

	cout<<"Parameters:"<<endl;
	cout<<"Header's rows: "<<headerRows<<endl;
	cout<<"Label columns: "<<labelColumns<<endl;
	cout<<"Epsilon: "<<epsilon<<endl;
	cout<<"Iterations to wait: "<<tWait<<endl<<endl;
	cout<<"Reading file: "<<filePath<<endl;
	vector<vector<string>> dataRaw=readData(filePath,headerRows,delimiter);
	vector<string> row=dataRaw[0];
	int attributeSize=row.size()-labelColumns;
	int dataNum=dataRaw.size();
	cout<<"Data read properly"<<endl;
	cout<<"Objects in data set: "<<dataNum<<endl<<endl;

	/**
	 * Converting vector of data to the float array
	 * with additional columns for length and cluster ID
	 */

	float data[dataNum][attributeSize+2];
	for(int i=0; i<dataNum; i++){
		float dataVector[attributeSize];
		for(int ii=0; ii<attributeSize; ii++){
			dataVector[ii]=stof(dataRaw[i][ii+labelColumns]);
			data[i][ii]=dataVector[ii];
		}
		data[i][attributeSize]=vectorLength(dataVector,attributeSize);
		data[i][attributeSize+1]=i;
		/**
		//Normalize vector
		float vectorLengthTmp=vectorLength(dataVector);
		float normalizedVector[attributeSize];
		for(int ii=0; ii<attributeSize; ii++){
			data[i][ii]=dataVector[ii]/vectorLengthTmp;
			normalizedVector[ii]=data[i][ii];
		}
		data[i][attributeSize]=vectorLength(normalizedVector);//Just in case
		**/
	}
	//At the end -> 2D array of vectors + length + claster ID

	/**
	 * Sorting data by vector length
	 */

	//Sorting
	dataVec dataArray[dataNum];
	for(int i=0; i<dataNum;i++){
		dataArray[i]={data[i],data[i][attributeSize]};
	}
	int n = sizeof(dataArray)/sizeof(dataArray[0]);
	sort(dataArray, dataArray+n, compareLength);

	//Writing sorted data to tmp array
	float dataTmpArray[dataNum][attributeSize+1];
	for(int i=0; i<dataNum; i++){
		for(int ii=0; ii<attributeSize+1; ii++){
			dataTmpArray[i][ii]=dataArray[i].vector[ii];
		}
	}
	//Copying data to the "main" array
	for(int i=0; i<dataNum; i++){
		for(int ii=0; ii<attributeSize+1; ii++){
			data[i][ii]=dataTmpArray[i][ii];
		}
	}
	//At the end -> sorted 2D data float array

	/**
	 * KNN clasterization using cosine measure
	 */

	cout<<"Starting clasterization process"<<endl;
	uint16_t clusterNumPrev=0;
	int algorithmStopCounter=0;
	//Main algorithm's loop
	while(true){
		//Creating tmp (working) array
		float dataTmp[dataNum][attributeSize+2];
		for (int i=0;i<dataNum;i++){
			for (int ii=0; ii<attributeSize+2;ii++){
				dataTmp[i][ii]=data[i][ii];
			}
		}

		//Iteration through each object
		//Finding it's neighbors and setting it's cluster ID
		for (int i=0; i<dataNum; i++){
			//Calculate boundary values for potential close vectors
			float minBound=epsilon*data[i][attributeSize];
			float maxBound=data[i][attributeSize]/epsilon;

			//Search for potential close vectors and save their IDs
			vector<int> closeCandidates;
			//To save the memory/time iterate from ith element in + and - directions
			for (int ii=i-1;ii>=0;ii--){
				if(data[ii][attributeSize]>=minBound){
					closeCandidates.push_back(ii);
				}
				else{
					break;
				}
			}
			for (int ii=i+1;ii<dataNum;ii++){
				if(data[ii][attributeSize]<=maxBound){
					closeCandidates.push_back(ii);
				}
				else{
					break;
				}
			}
			//If there's no candidates move to another object
			if(closeCandidates.size()<1)
				continue;

			//Analyzing the cosine measure of potential close vectors to determine if they're close ones
			vector<int> closeVectors;
			for (uint16_t ii=0;ii<closeCandidates.size();ii++){
				float product=0;
				for (int iii=0;iii<attributeSize;iii++){
					product+=data[i][iii]*data[closeCandidates[ii]][iii];
				}
				float cosine=product/(data[i][attributeSize]*data[closeCandidates[ii]][attributeSize]);
				if (cosine>=epsilon){
					closeVectors.push_back(closeCandidates[ii]);
				}
			}

			//If there's no close vectors
			if(closeVectors.size()<1)
				continue;

			//KNN on vectors
			vector<clusterInfo> clusterSizes;
			for (uint16_t ii=0; ii<closeVectors.size(); ii++){
				//Check if cluster is already in neighborhood
				bool notNeighbour=true;
				for (uint16_t iii=0;iii<clusterSizes.size();iii++){
					if(clusterSizes[iii].id==data[closeVectors[ii]][attributeSize+1]){
						clusterSizes[iii].size++;
						notNeighbour=false;
					}
				}
				//If not, add to vector new cluster
				if(notNeighbour){
					clusterInfo cluster={(int)data[closeVectors[ii]][attributeSize+1],1};
					clusterSizes.push_back(cluster);
				}
			}
			sort(clusterSizes.begin(),clusterSizes.end(),compareClusters);
			dataTmp[i][attributeSize+1]=clusterSizes[0].id;
		}
		for (int i=0;i<dataNum;i++){
			for (int ii=0; ii<attributeSize+2;ii++){
				data[i][ii]=dataTmp[i][ii];
			}
		}

		//Count clasters sizes
		vector<int> clasterSizes;
		//Prepare clusters id array
		float clusterArray[dataNum];
		for (int ii=0; ii<dataNum; ii++){
			clusterArray[ii]=data[ii][attributeSize+1];
		}
		//Calculate cluster sizes
		for (int ii=0; ii<dataNum; ii++){
			int clasterSize=countItem(ii, clusterArray, dataNum);
			if (clasterSize>0)
				clasterSizes.push_back(clasterSize);
		}
		//cout<<clasterSizes.size()<<endl;

		//If number of clusters doesn't change, end algorithm
		if (clasterSizes.size()==clusterNumPrev){
			if(algorithmStopCounter>tWait){
				break;
			}
			else{
				algorithmStopCounter++;
			}
		}
		else{
			clusterNumPrev=clasterSizes.size();
			algorithmStopCounter=0;
		}
	}

	/*************************
	Preparing data for saving:
	Sorting by cluster
	Changing cluster ids to from 0
	*************************/

	//Sort data by cluster
	dataVec idAndCluster[dataNum]; //Object + cluster ID (as length, cause there's no need to create new class and functions)
	for (int i=0;i<dataNum;i++){
		idAndCluster[i].vector=data[i];
		idAndCluster[i].length=data[i][attributeSize+1];
	}
	n = sizeof(idAndCluster)/sizeof(idAndCluster[0]);
	sort(idAndCluster,idAndCluster+n,compareLength);

	//Writing sorted data to tmp array
	float dataTmp2Array[dataNum][attributeSize+2];
	for(int i=0; i<dataNum; i++){
		for(int ii=0; ii<attributeSize+2; ii++){
			dataTmp2Array[i][ii]=idAndCluster[i].vector[ii];
		}
	}
	int currentOldID=-1;
	int currentNewID=-1;
	//Copying data to the "main" array
	//Plus changing the cluster ids (0,1,2,...)
	for(int i=0; i<dataNum; i++){
		if (currentOldID!=dataTmp2Array[i][attributeSize+1]){
			currentOldID=dataTmp2Array[i][attributeSize+1];
			currentNewID++;
		}
		for(int ii=0; ii<attributeSize+1; ii++){
			data[i][ii]=dataTmp2Array[i][ii];
		}
		data[i][attributeSize+1]=currentNewID;
	}
	cout<<"Founded clasters: "<<data[dataNum-1][attributeSize+1]+1<<endl<<endl;

	/*************************
	Saving data to the new csv file
	*************************/

	float **dataTmp=new float *[dataNum];
	for(int i = 0; i<dataNum; i++){
	    dataTmp[i] = new float[attributeSize+1];
	    for (int ii=0; ii<attributeSize+2; ii++){
	    	dataTmp[i][ii]=data[i][ii];
	    }
	}
	writeData(dataTmp,dataNum,attributeSize+2,"E:\\Users\\Polonius\\Downloads\\data.csv");
	cout<<"Data Saved"<<endl;

	return 0;
}
