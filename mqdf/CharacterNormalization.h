#ifndef CHARACTER_NORMALIZATION_H_
#define CHARACTER_NORMALIZATION_H_


namespace cv {
class Mat;
}

namespace ljocr {

class CharacterNormalization {
  public:
    CharacterNormalization(int normalized_size);
    ~CharacterNormalization();
    int normalizebyBiMoment(cv::Mat& src_character, cv::Mat& normal_character);
    int normalizebyARAN(cv::Mat& src_character, cv::Mat& normal_character);
    void normalizeGrayscale(cv::Mat& src, float m0, float sigma0, cv::Mat& pData);
  protected:
  private:
    int normalized_size_;
};

}
#endif