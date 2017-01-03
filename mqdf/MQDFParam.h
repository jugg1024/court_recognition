#ifndef MQDFPARAM_
#define MQDFPARAM_

#include "opencv2/highgui/highgui.hpp"

namespace ljocr {

struct MQDFParam {
    double remainder;
    double delta;
    int principle_dim;
    cv::Mat principle_eigenvalue;
    cv::Mat principle_eigenvector;
    cv::Mat mean;
    cv::Mat cov;
};

}
#endif