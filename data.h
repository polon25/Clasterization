#ifndef DATA_H_
#define DATA_H_

#include <iostream>
#include <vector>

//Read data from given file
std::vector<std::vector<std::string>> readData(std::string fileName,int firstLine=0,char delimiter=',');
void writeData(float **data,int col, int row,std::string fileName);

#endif /* DATA_H_ */
