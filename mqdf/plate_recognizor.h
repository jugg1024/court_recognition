#ifndef PLATE_RECOGNIZOR_H
#define PLATE_RECOGNIZOR_H
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
using namespace std;

namespace cv {
class Mat;
}

class PlateRecognizor {
public:
  // construct and deconstruct
  PlateRecognizor(std::string mqdf_model_path);
  ~PlateRecognizor();

  // recognit
  std::string recognit(std::string im_path);
  // set the model of MQDF classifier
  void SetModel(std::string mqdf_model_path);
  // all court plate names
  std::vector<std::string> dict;
  std::string model_path;
};
#endif


	