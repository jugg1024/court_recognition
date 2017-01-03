#include <opencv2/opencv.hpp>
#include "LSMQDFClassification.h"

using namespace cv;

std::map<double, std::string, std::greater<double> > NewSample(int CFE_block, float rank, Mat test_img, std::string mqdf_load_path);
