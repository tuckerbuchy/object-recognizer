#include "extractor.h"

int main(int argc, char ** argv) {

	ExtractHelper::performClusteringSteps();

	cout<<"Processing training data..."<<endl;
	Mat trainingData(0, ExtractHelper::dictionarySize, CV_32FC1);
	Mat labels(0, 1, CV_32FC1);

	ExtractHelper::extractBOWDescriptor(ExtractHelper::TRAINING_PATH, trainingData,labels);

	NormalBayesClassifier classifier;
	cout<<"Training classifier..."<<endl;

	classifier.train(trainingData, labels);

	//save the classifier to be loaded during evaluation.
	classifier.save("../xml/trainer_snapshot.xml");
}
