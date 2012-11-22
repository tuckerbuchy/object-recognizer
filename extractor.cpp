#include "extractor.h"

void ExtractHelper::performClusteringSteps()
{
	cout<<"Creating dictionary..."<<endl;
	extractTrainingVocabulary(TRAINING_PATH);
	vector<Mat> descriptors = bowTrainer.getDescriptors();
	int count=0;
	for(vector<Mat>::iterator iter=descriptors.begin();iter!=descriptors.end();iter++)
	{
		count+=iter->rows;
	}
	cout<<"Clustering "<<count<<" features"<<endl;
	Mat dictionary = bowTrainer.cluster();

	//save the clusters to be loaded during evaluation
	FileStorage fs("clusters_snapshot.xml", FileStorage::WRITE);
	fs << "dictionary" << dictionary;

	bowDescriptorExtractor.setVocabulary(dictionary);
}

bool ExtractHelper::loadCachedClusters()
{
	Mat dictionary(0, 0, CV_32FC1);

	//load the matrix from cached xml
	FileStorage fs("clusters_snapshot.xml", FileStorage::READ);
	fs["dictionary"] >> dictionary;

	if (dictionary.dims == 0)
	{
		cout<<"Did not correctly read the cluster matrix from memory. "
				"Are you sure clustering has occured?"<<endl;
		return false;
	}

	ExtractHelper::bowDescriptorExtractor.setVocabulary(dictionary);
	return true;
}

/**
 * \brief Recursively traverses a folder hierarchy. Extracts features from the training images and adds them to the bowTrainer.
 */
void ExtractHelper::extractTrainingVocabulary(const path& basepath) {

	for (directory_iterator iter = directory_iterator(basepath); iter
				!= directory_iterator(); iter++) {
			directory_entry entry = *iter;
			cout << "Processing directory " << entry.path().string() << endl;
			for (directory_iterator iter = directory_iterator(entry.path()); iter
						!= directory_iterator(); iter++) {
				path photoPath = *iter;
				if (photoPath.extension() == ".JPG"){
					cout << "Processing file " << photoPath.string() << endl;
					Mat img = imread(photoPath.string());
					if (!img.empty())
					{
						vector<KeyPoint> keypoints;
						detector->detect(img, keypoints);
						if (keypoints.empty())
						{
							cerr << "Warning: Could not find key points in image: "<< photoPath.string() << endl;
						}
						else
						{
							Mat features;
							extractor->compute(img, keypoints, features);
							bowTrainer.add(features);
						}
				}
					else
					{
					cout<<"Image not found."<<endl;
				}
			}
		}
	}
}


/**
 * \brief Recursively traverses a folder hierarchy. Creates a BoW descriptor for each image encountered.
 */
void ExtractHelper::extractBOWDescriptor(const path& basepath, Mat& descriptors, Mat& labels) {
	float count = 0;
	for (directory_iterator iter = directory_iterator(basepath); iter
			!= directory_iterator(); iter++) {
		directory_entry entry = *iter;
		count++;
		cout << "Processing directory " << entry.path().string() << endl;
		for (directory_iterator iter = directory_iterator(entry.path()); iter
					!= directory_iterator(); iter++) {
			path photoPath = *iter;
			if (photoPath.extension() == ".JPG"){
				cout << "Processing file " << photoPath.string() << endl;
				Mat img = imread(photoPath.string());
				if (!img.empty()) {
					vector<KeyPoint> keypoints;
					detector->detect(img, keypoints);
					if (keypoints.empty()) {
						cerr << "Warning: Could not find key points in image: "<< photoPath.string() << endl;
					} else {
						Mat bowDescriptor;
						bowDescriptorExtractor.compute(img, keypoints, bowDescriptor);
						descriptors.push_back(bowDescriptor);
						labels.push_back(count);
					}
				}
			}
		}
	}
}
