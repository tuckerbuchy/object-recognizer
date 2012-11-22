#ifndef EXTRACTOR_H
#define EXTRACTOR_H

#include <string>
#include <iostream>
#include <vector>
#include <boost/filesystem.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/ml.h>
#include <opencv2/nonfree/features2d.hpp>

using namespace std;
using namespace boost::filesystem;
using namespace cv;

namespace ExtractHelper{

    static path TRAINING_PATH = path("data/train/");
    static path EVAL_PATH = path("data/eval/");

	static Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
	static Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SURF");
	static Ptr<FeatureDetector> detector= new SurfFeatureDetector();

	static int dictionarySize = 150;

	static TermCriteria tc(CV_TERMCRIT_ITER, 10, 0.001);

	static int retries = 1;

	static BOWKMeansTrainer bowTrainer(dictionarySize, tc, retries);
	static BOWImgDescriptorExtractor bowDescriptorExtractor(extractor, matcher);

	void performClusteringSteps();

	bool loadCachedClusters();

	void extractTrainingVocabulary(const path& basepath);

	void extractBOWDescriptor(const path& basepath, Mat& descriptors, Mat& labels);
};
#endif
