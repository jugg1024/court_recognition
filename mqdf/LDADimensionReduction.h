#ifndef LDA_DIMENSION_REDUCTION_H_
#define LDA_DIMENSION_REDUCTION_H_

#include<opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
//#include "contrib/contrib.hpp"

#include <vector>

namespace ljocr {

using std::vector;
using std::string;
using std::map;

class LDADimensionReduction {
  public:
    LDADimensionReduction(int sample_dim);
    ~LDADimensionReduction();

    int addSamples(std::string character, vector<vector<float> >& samples);
    int train(int dimension);
    int save(std::string lda_param_path);

    int load(std::string lda_param_path);
    int reduceDimension(std::vector<float>& feature, std::vector<float>& out_feature);
  protected:
  private:
    std::vector<std::pair<int, cv::Mat> > info_k_;
    cv::Mat sw_;
    cv::Mat mean_;
    cv::Mat sb_;
    int num_;
    
    cv::Mat eigenvalues_;
    cv::Mat eigenvectors_;
};

//lj debug
//int test1(std::string character, vector<vector<float> >& samples);
//int test2(std::string character, vector<vector<float> >& samples);
//lj end
}

#endif  //LDA_DIMENSION_REDUCTION_H_