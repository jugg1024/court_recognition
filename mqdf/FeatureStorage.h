#ifndef FEATURE_STORAGE_H_
#define FEATURE_STORAGE_H_

#include <string>
#include <vector>

namespace cv {
class Mat;
}
namespace ljocr {

class FeatureStorage {
  public:
    FeatureStorage(int mode);
    ~FeatureStorage();

    int save(const std::string& filename, std::vector<std::pair<std::string, cv::Mat> >& feature);
    int load(const std::string& filename, std::vector<std::pair<std::string, cv::Mat> >& feature);
  private:
    int mode_; //0 represent YAML, 1 represent binary
};

}

#endif