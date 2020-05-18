#include "data.h"
#include <fstream>
#include <sstream>

using namespace std;

//Read data from given file
vector<vector<string>> readData(string fileName,int firstLine,char delimiter){
	//Open data file
	fstream fin;
	fin.open(fileName, ios::in);

	//Data stored in 2d vector of strings
	vector<vector<string>> data;
	vector<string> row;
	string line,word,temp;
	//Read data
	int i=0;
	while(getline(fin,line)){
		row.clear();
		//Read line and store in 'line'
		//Don't read first n lines
		if (i<firstLine){
			i++;
			continue;
		}
		//Break words
		stringstream s(line);
		//Read every column and store in in 'word;
		while(getline(s,word,delimiter)){
			row.push_back(word);
		}
		//Append row to the data vector
		data.push_back(row);
	}
	//Close file
	fin.close();
	return data;
}

//Save data as csv file
void writeData(float **data,int col, int row,string fileName){
	//Open/create a file
	fstream fout;
	fout.open(fileName,ios::out);

	for(int i=0; i<col;i++){
		for(int ii=0; ii<row;ii++){
			fout<<data[i][ii];
			if(ii<row-1)
				fout<<',';
		}
		fout<<'\n';
	}
	fout.close();
}
