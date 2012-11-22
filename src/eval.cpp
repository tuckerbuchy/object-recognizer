#include "extractor.h"

using namespace std;
using namespace boost::filesystem;
using namespace cv;

int main(int argc, char ** argv) {

	bool loadedSuccessfully =  ExtractHelper::loadCachedClusters();
	if (!loadedSuccessfully)
		return -1;

	NormalBayesClassifier classifier;
	classifier.load("../xml/trainer_snapshot.xml");


	cout<<"Processing evaluation data..."<<endl;
	Mat evalData(0, ExtractHelper::dictionarySize, CV_32FC1);
	Mat groundTruth(0, 1, CV_32FC1);
	ExtractHelper::extractBOWDescriptor(ExtractHelper::EVAL_PATH, evalData, groundTruth);

	cout<<"Evaluating classifier..."<<endl;
	Mat results;
	classifier.predict(evalData, &results);

	for (int i = 0; i < results.cols; i++){
		for(int j = 0; i < results.rows; i++){
			cout<<results.at<float>(i,j)<<endl;
		}
	}

	double errorRate = (double) countNonZero(groundTruth - results) / evalData.rows;
	cout << "Error rate: " << errorRate << endl;
}
