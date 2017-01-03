//#include <io.h>
#include <stdio.h>
#include "opencv2/highgui/highgui.hpp"
#include "CharacterFeatureExtraction.h"
#include "LSMQDFClassification.h"
#include "CharacterNormalization.h"
#include "FeatureStorage.h"
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
using namespace ljocr;
using namespace cv;
using namespace std;



//float NewSample(int CFE_block, float rank, string test_path, string lda_path, string mqdf_load_path)
std::map<double, std::string, std::greater<double> > NewSample(int CFE_block, float rank, Mat test_img, std::string mqdf_load_path)
{
	if (test_img.channels()>1)
		cvtColor(test_img, test_img, CV_BGR2GRAY);
	

	CharacterFeatureExtraction fe(8, CFE_block);
	
	LSMQDFClassification lsmqdfcl;
	CharacterNormalization normal(64);



	FeatureStorage fs(0);

	lsmqdfcl.loadTemplate(mqdf_load_path);
	int correct_num = 0;
	float sum = 0;
	int sum2 = 0;

	int num = 0;

	vector< vector<float> > features;



	vector<float> feature;
	Mat character;
	//normal.normalizebyARAN(test_img, character);
	normal.normalizebyBiMoment(test_img, character);
	fe.extractFeaturebyHOG(character, 1, feature);
	//lj Box-cox begin
	for (vector<float>::iterator iter2 = feature.begin(); iter2 != feature.end(); ++iter2) {
		*iter2 = sqrt(*iter2);
	}
	//lj Box-cox end
	//vector<float> out_feature;
	//lda.reduceDimension(feature, out_feature);
	std::string character_str;
	std::map<double, std::string, std::greater<double> > recognition_result;
	//double confidence = mqdfcl.recognize(out_feature,1, recognition_result);
	//out_feature.clear();
	//double confidence = mqdfcl.recognize(feature,3, recognition_result);
	double confidence = lsmqdfcl.recognize(feature, rank, recognition_result);
	feature.clear();



	return recognition_result;


	//return recognition_result.rbegin()->second;
}

