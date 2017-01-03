#ifndef CHARACTER_FEATURE_EXTRACTION_H
#define CHARACTER_FEATURE_EXTRACTION_H

#include <vector>

namespace cv {
class Mat;
}

namespace ljocr {

class CharacterFeatureExtraction {
  public:
    CharacterFeatureExtraction(int num_direction, int num_block);
    ~CharacterFeatureExtraction();
    int extractFeaturebyHOG(const cv::Mat& character, const int background, std::vector<float>& feature);  //if background == 0, then background is black, else it is white
  protected:
  private:
    int num_direction_;
    int num_block_;

};
}
#endif