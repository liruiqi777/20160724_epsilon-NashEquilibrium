/*
 * main.cpp
 *
 *  Created on: Jul 6, 2016
 *      Author: RLi
 */


#include <fstream>
#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <stack>
#include <iterator>

#define DATAPOINTS 14

using namespace std;

int main(){

	string fileName;
	ifstream dataFile;

	ofstream file;
	file.open("Data Summary.csv");

	do{
		cout << "Enter a csv file: ";
		cin >> fileName;
		dataFile.open(fileName.c_str(), ios::in);
	} while(!dataFile.good());

	string line;

	if (!getline(dataFile, line)){
		cerr << "The csv file is empty." << endl;
		return EXIT_FAILURE;
	}

	int gameNum = 0;
	int solutionNum = 0;

	while (getline(dataFile, line)){

		istringstream lineStream(line);
		string data;
		getline(lineStream, data, ',');

		if (stoi(data) != gameNum && gameNum != 0){

			file << solutionNum << endl;
			gameNum = stoi(data);
			file << gameNum << ",";
			for (int i = 1; i < 7; i++){
				getline(lineStream, data, ',');
				file << data << ",";
			}

			solutionNum = 1;

		} else if (gameNum == 0){
			gameNum = stoi(data);
			file << gameNum << ",";
			for (int i = 1; i < 7; i++){
				getline(lineStream, data, ',');
				file << data << ",";
			}
			solutionNum++;
		}else{
			solutionNum++;
		}
	}

	file << solutionNum << endl;

}
