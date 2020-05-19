//============================================================================
// Name        : main.cpp
// Author      : Jacek Pi³ka
// Description : DAMI Project -> clusterization using cosine measure
// Arguments: filePath, number of header rows, number of columns with labels, delimiter, epsilon, k, filePathOut
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
	int k=atoi(arg[6]); //k neighbors
	string filePathOut=arg[7];

	/**
	 * Importing data from file
	 */

	cout<<"Parameters:"<<endl;
	cout<<"Header's rows: "<<headerRows<<endl;
	cout<<"Label columns: "<<labelColumns<<endl;
	cout<<"Epsilon: "<<epsilon<<endl;
	cout<<"k:"<<k<<endl<<endl;
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
		data[i][attributeSize+1]=-1; //Initialize claster ID value with -1
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
	int currentClusterID=0;
	//Main algorithm's loop

	//Iteration through each object
	//Finding it's neighbors and setting it's cluster ID
	for (int i=0; i<dataNum; i++){
		//Calculate boundary values for potential close vectors
		float minBound=epsilon*data[i][attributeSize];
		float maxBound=data[i][attributeSize]/epsilon;

		/**
		 * Analyzing cosine values of potential close vectors
		 * Two for loops to iterate from ith vector in both directions
		 */

		vector<objectInfo> closeVectors; //IDs of close vectors
		for (int ii=i-1;ii>=0;ii--){
			if(data[ii][attributeSize]>=minBound){//If potential close vector
				//Calculate scalar product of vectors
				float product=0;
				for (int iii=0;iii<attributeSize;iii++){
					product+=data[i][iii]*data[ii][iii];
				}
				float cosine=product/(data[i][attributeSize]*data[ii][attributeSize]);
				if (cosine>=epsilon){//If cosine value is enough iith vector is close
					objectInfo object; object.id=ii; object.cosineVal=cosine;
					closeVectors.push_back(object);
				}
			}
			else{//Break loop if we exceed boundary
				break;
			}
		}
		for (int ii=i+1;ii<dataNum;ii++){
			if(data[ii][attributeSize]<=maxBound){//If potential close vector
				//Calculate scalar product of vectors
				float product=0;
				for (int iii=0;iii<attributeSize;iii++){
					product+=data[i][iii]*data[ii][iii];
				}
				float cosine=product/(data[i][attributeSize]*data[ii][attributeSize]);
				if (cosine>=epsilon){//If cosine value is enough iith vector is close
					objectInfo object; object.id=ii; object.cosineVal=cosine;
					closeVectors.push_back(object);
				}
			}
			else{//Break loop if we exceed boundary
				break;
			}
		}

		//If there's no close vectors, move to next object
		if((int)closeVectors.size()<k)
			continue;

		//Sort neighboors by their cosine value
		objectInfo idAndCosine[closeVectors.size()]; //Object + cluster ID (as length, cause there's no need to create new class and functions)
		for (int ii=0;ii<(int)closeVectors.size();ii++){
			idAndCosine[ii].id=closeVectors[ii].id;
			idAndCosine[ii].cosineVal=closeVectors[ii].cosineVal;
		}
		int n = sizeof(idAndCosine)/sizeof(idAndCosine[0]);
		sort(idAndCosine, idAndCosine+n, compareObjects);

		/**
		 * Clusterization:
		 *
		 * 1. Check if any of k close vectors is already in cluster
		 * 2. If yes, than the ith object will be added to this cluster alongside other non clustered objects
		 * 3. If no, create new cluster and add every neighbor
		 */

		//Check if any of k close vectors is already in cluster
		int currentClusterIDTmp=currentClusterID;
		for(int ii=0; ii<k;ii++){
			if (data[idAndCosine[ii].id][attributeSize+1]>=0){
				currentClusterIDTmp=data[idAndCosine[ii].id][attributeSize+1];
				break;
			}
		}

		//Add ith object to the cluster alongside other non clustered objects
		data[i][attributeSize+1]=currentClusterIDTmp;
		for(int ii=0; ii<k;ii++){
			if (data[idAndCosine[ii].id][attributeSize+1]<0){
				data[idAndCosine[ii].id][attributeSize+1]=currentClusterIDTmp;
			}
		}

		//If new cluster was create, increase the future cluster ID by 1
		if(currentClusterIDTmp==currentClusterID){
			currentClusterID++;
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

	float **dataTmpSave=new float *[dataNum];
	for(int i = 0; i<dataNum; i++){
		dataTmpSave[i] = new float[attributeSize+1];
	    for (int ii=0; ii<attributeSize+2; ii++){
	    	dataTmpSave[i][ii]=data[i][ii];
	    }
	}
	writeData(dataTmpSave,dataNum,attributeSize+2,filePathOut);
	cout<<"Data Saved"<<endl;

	return 0;
}
