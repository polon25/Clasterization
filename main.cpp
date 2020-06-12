//============================================================================
// Name        : main.cpp
// Author      : Jacek Pi³ka
// Description : DAMI Project -> clusterization using cosine measure
// Arguments: filePath, number of header rows, number of columns with labels, delimiter, epsilon, k, filePathOut
// Beta version for testing new ideas
//============================================================================

#include "data.h"
#include "cosine.h"
#include <algorithm> //For sort()
#include <chrono>
#include <ctime>

using namespace std;

int main (int argc, char *arg[]){
	//Start counting
	auto start = std::chrono::system_clock::now();

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
	float borderAngle=acos(epsilon);
	float borderEuclidean=sqrt(2-2*epsilon);
	cout<<"Angle border: "<<borderAngle<<endl;
	cout<<"Euclidean border: "<<borderEuclidean<<endl;
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

	float data[dataNum][attributeSize+3];
	float dataOld[dataNum][attributeSize+3];
	for(int i=0; i<dataNum; i++){
		float dataVector[attributeSize];
		for(int ii=0; ii<attributeSize; ii++){
			dataVector[ii]=stof(dataRaw[i][ii+labelColumns]);
			data[i][ii]=dataVector[ii];
			dataOld[i][ii]=dataVector[ii];
		}
		data[i][attributeSize]=vectorLength(dataVector,attributeSize);
		data[i][attributeSize+1]=-1; //Initialize claster ID value with -1
		data[i][attributeSize+2]=i; //Last cell -> original id
		dataOld[i][attributeSize]=data[i][attributeSize];
		dataOld[i][attributeSize+1]=-1; //Initialize claster ID value with -1
		dataOld[i][attributeSize+2]=i; //Last cell -> original id
	}

	//Normalize vector and calculate angle from vector [1,0,0,...,0]
	for(int i=0; i<dataNum; i++){
		for(int ii=0; ii<attributeSize; ii++){
			data[i][ii]=data[i][ii]/data[i][attributeSize];
		}

		//Calculate angle from arbitrary vector, which will be the indicator of potential close vectors
		data[i][attributeSize]=acos(data[i][0]);

		//Commentary to this method (because it's a bigger thinking wrapped into one line of code)
		//First we calculate the scalar product of two vectors, but because the second has only one nonzero part and it's equal 1,
		//it will be equal simply to respective part of first vector.
		//a x b = |a| |b| cos(alpha), but because they're both normalized, cos(alpha)=a x b = a[0]
		//Than the angle in radians is calculated using acos

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
	sort(dataArray, dataArray+n, compareAngle);

	//Writing sorted data to tmp array
	float dataTmpArray[dataNum][attributeSize+3];
	for(int i=0; i<dataNum; i++){
		for(int ii=0; ii<attributeSize+3; ii++){
			dataTmpArray[i][ii]=dataArray[i].vector[ii];
		}
	}
	//Copying data to the "main" array
	for(int i=0; i<dataNum; i++){
		for(int ii=0; ii<attributeSize+3; ii++){
			data[i][ii]=dataTmpArray[i][ii];
		}
	}
	//At the end -> sorted 2D data float array

	cout<<"Create matrix"<<endl;

	//Matrix of neighbors (0 - not neighbor, 1 - neighbor)
	//First id - neighbors of i-th object, Second id - objects which have i-th element as their neighbor
	vector<int*> neighborMatrix; //Cause it's to big array to do it the "normal way"

	for (int i=0; i<dataNum; i++){
		neighborMatrix.push_back(new int[dataNum]);
		for (int ii=0; ii<dataNum; ii++){
			neighborMatrix[i][ii]=0;
		}
	}

	cout<<"Matrix created"<<endl;
	int objectClassTable[dataNum]={NOISE};

	/**
	 * k+ NBC clusterization using cosine measure
	 */

	cout<<"Starting clusterization process"<<endl;
	int currentClusterID=0;
	//Main algorithm's loop

	//First main loop -> searching for neighbors and making neighborMatrix
	for (int i=0; i<dataNum; i++){
		//Calculate boundary values for potential close vectors
		float minAngle=data[i][attributeSize]-borderAngle;
		float maxAngle=data[i][attributeSize]+borderAngle;

		/**
		 * Analyzing cosine values of potential close vectors
		 * Two for loops to iterate from ith vector in both directions
		 */

		vector<objectInfo> closeVectors; //IDs of close vectors
		for (int ii=i-1;ii>=0;ii--){
			if(data[ii][attributeSize]>=minAngle){//If potential close vector
				//Calculate Euclidean Distance
				float sum=0;
				for (int iii=0;iii<attributeSize;iii++){
					float a=dataOld[(int)data[i][attributeSize+2]][iii]-dataOld[(int)data[ii][attributeSize+2]][iii];
					sum+=a*a;
				}
				sum=sqrt(sum)/dataOld[(int)data[i][attributeSize+2]][attributeSize];
				if (sum<=borderEuclidean){//Check if object is close enough
					objectInfo object; object.id=ii; object.euclideanDistance=sum;
					closeVectors.push_back(object);
				}
			}
			else{//Break loop if we exceed boundary
				break;
			}
		}
		for (int ii=i+1;ii<dataNum;ii++){
			if(data[ii][attributeSize]<=maxAngle){//If potential close vector
				//Calculate Euclidean Distance
				float sum=0;
				for (int iii=0;iii<attributeSize;iii++){
					float a=dataOld[(int)data[i][attributeSize+2]][iii]-dataOld[(int)data[ii][attributeSize+2]][iii];
					sum+=a*a;
				}
				sum=sqrt(sum)/dataOld[(int)data[i][attributeSize+2]][attributeSize];
				if (sum<=borderEuclidean){//Check if object is close enough
					objectInfo object; object.id=ii; object.euclideanDistance=sum;
					closeVectors.push_back(object);
				}
			}
			else{//Break loop if we exceed boundary
				break;
			}
		}

		//Sort neighboors by their euclidean distance
		objectInfo idAndCosine[closeVectors.size()]; //Object + cluster ID (as length, cause there's no need to create new class and functions)
		for (int ii=0;ii<(int)closeVectors.size();ii++){
			idAndCosine[ii].id=closeVectors[ii].id;
			idAndCosine[ii].euclideanDistance=closeVectors[ii].euclideanDistance;
		}
		int n = sizeof(idAndCosine)/sizeof(idAndCosine[0]);
		sort(idAndCosine, idAndCosine+n, compareObjects);

		//If less neighbors then k
		int kTmp=(k<n)? k : n;

		//Add k+ neighbors to neighborMatrix
		for(int ii=0; ii<n; ii++){
			if (idAndCosine[ii].euclideanDistance<=idAndCosine[kTmp-1].euclideanDistance){ //Check if in k+ neigborhood
				neighborMatrix[i][idAndCosine[ii].id]=1; //Add to neighborMatrix
			}
			else
				break;
		}
	}

	cout<<"Neighbor matrix completed"<<endl;
	cout<<"Starting cluster searching"<<endl;

	//Second main loop -> clustering objects
	//Only core points can initialize cluster, border points can be only added to already existing one
	for (int i=0; i<dataNum; i++){
		double kn=0; double rkn=0;//k-neighoors and reversed k-neighoors

		vector<int> neigborsIDs;

		for (int ii=0; ii<dataNum; ii++){
			kn+=neighborMatrix[i][ii];
			rkn+=neighborMatrix[ii][i];
			if (neighborMatrix[i][ii]>0){
				neigborsIDs.push_back(ii);
			}
		}

		//Check the class of object
		//Only cores can create clusters (but borders can be added to them)

		double ndf=(rkn>0)? kn/rkn : 0;

		if ((kn>=k) & (ndf>=1))
			objectClassTable[i]=CORE;
		else if ((kn>=k) & (ndf<1)){
			objectClassTable[i]=BORDER;
			continue;
		}
		else{
			objectClassTable[i]=NOISE;
			continue;
		}

		/**
		 * Clusterization:
		 *
		 * 1. Check if any of k close vectors is already in cluster
		 * 2. If yes, than the ith object will be added to this cluster alongside other non clustered objects
		 * 3. If no, create new cluster and add every neighbor
		 */

		//Check if any of k close vectors is already in cluster
		int currentClusterIDTmp=currentClusterID;
		for(int ii=0; ii<(int)neigborsIDs.size();ii++){
			if (data[neigborsIDs[ii]][attributeSize+1]>=0){
				currentClusterIDTmp=data[neigborsIDs[ii]][attributeSize+1];
				break;
			}
		}

		//Add ith object to the cluster alongside other non clustered objects
		data[i][attributeSize+1]=currentClusterIDTmp;
		for(int ii=0; ii<(int)neigborsIDs.size();ii++){
			if (data[neigborsIDs[ii]][attributeSize+1]<0){
				data[neigborsIDs[ii]][attributeSize+1]=currentClusterIDTmp;
			}
		}

		//If new cluster was create, increase the future cluster ID by 1
		if(currentClusterIDTmp==currentClusterID){
			currentClusterID++;
		}
	}

	cout<<"Clasterization Complete"<<endl;

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
	sort(idAndCluster,idAndCluster+n,compareAngle);

	//Writing sorted data to tmp array
	float dataTmp2Array[dataNum][attributeSize+3];
	for(int i=0; i<dataNum; i++){
		for(int ii=0; ii<attributeSize+3; ii++){
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
		for(int ii=0; ii<attributeSize+3; ii++){
			data[i][ii]=dataTmp2Array[i][ii];
		}
		data[i][attributeSize+1]=currentNewID;
	}
	cout<<"Founded clusters: "<<data[dataNum-1][attributeSize+1]+1<<endl<<endl;

	/*************************
	Saving data to the new csv file
	*************************/

	float **dataTmpSave=new float *[dataNum];
	for(int i = 0; i<dataNum; i++){
		dataTmpSave[i] = new float[attributeSize+2];
	    for (int ii=0; ii<attributeSize+1; ii++){
	    	//dataTmpSave[i][ii]=data[i][ii];
	    	dataTmpSave[i][ii]=dataOld[(int)data[i][attributeSize+2]][ii];
	    }
	    dataTmpSave[i][attributeSize+1]=data[i][attributeSize+1];
	}
	writeData(dataTmpSave,dataNum,attributeSize+2,filePathOut);
	cout<<"Data Saved"<<endl;

	auto end = std::chrono::system_clock::now();
	chrono::duration<double> elapsed_seconds = end-start;

	cout<<"elapsed time: " << elapsed_seconds.count() << "s\n";

	return 0;
}
