#ifndef COMMON_TOOLS_H_
#define COMMON_TOOLS_H_

namespace ljocr {
template <class NumType> cv::Mat Vect2MatEx(std::vector<std::vector<NumType> >& vect) {
    cv::Mat mtx = cv::Mat::zeros(vect.size(), vect[0].size(), cv::DataType<NumType>::type);
    for (int r = 0; r < vect.size(); r++) {
        for (int c = 0; c < vect[r].size(); c++) {
            mtx.at<NumType>(r, c) = vect[r][c];
        }
    }
    return mtx;
}

template <class NumType> cv::Mat Vect2Mat(std::vector<NumType>& vect) {
    cv::Mat mtx = cv::Mat::zeros(1, vect.size(), cv::DataType<NumType>::type);
    for (int c = 0; c < vect.size(); c++) {
        mtx.at<NumType>(0, c) = vect[c];
    }
    return mtx;
}

}

#endif  //COMMON_TOOLS_H_