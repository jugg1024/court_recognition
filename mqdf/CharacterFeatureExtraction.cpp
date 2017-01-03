#include "CharacterFeatureExtraction.h"

//#include <io>
#include <stdio.h>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace ljocr {

using cv::Mat;
using std::vector;

ljocr::CharacterFeatureExtraction::CharacterFeatureExtraction(int num_direction, int num_block) {
    num_direction_ = num_direction;
    num_block_ = num_block;
}

ljocr::CharacterFeatureExtraction::~CharacterFeatureExtraction() {

}

//	Standardize the gray scale level
static void normalizeGrayscale(const Mat& src, float m0, float sigma0, int background, Mat& pData) {
    pData = src.clone();
    Mat src2 = src.clone();
    int cols = pData.cols;
    int rows = pData.rows;
    if (pData.isContinuous()) {
        cols = pData.cols * pData.rows;
        rows = 1;
    }
    if (background) {
        for (int k = 0; k < rows; k++) {
            unsigned char* data = src2.ptr<uchar>(k);
            for (int i = 0; i < cols; ++i) {
                *(data + i) = 255 - *(data + i); //reverse image when the background is white
            }
        }
    }
    //eliminate background noise using OTSU
    cv::threshold(src2, pData, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    for (int k = 0; k < rows; k++) {
        unsigned char* data = pData.ptr<uchar>(k);
        unsigned char* src_data = src2.ptr<uchar>(k);
        for (int i = 0; i < cols; ++i) {
            //if (*(data + i) == 0) { //when the background is white
            if (*(data + i) > 0) {
                *(data + i) = *(src_data + i);
            }
        }
    }
    //end
    int nNum, k;
    float gSum;
    gSum = nNum = 0;
    for (k = 0; k < rows; k++) {
        unsigned char* data = pData.ptr<uchar>(k);
        for (int i = 0; i < cols; ++i) {
            //*(data + i) = 255 - *(data + i); //reverse image when the background is white
            if (*(data + i) > 0) {
                gSum += *(data + i);
                nNum++;
            }
        }
    }
    float mean = gSum / nNum;
    float sigma = 0;
    for (k = 0; k < rows; k++) {
        unsigned char* data = pData.ptr<uchar>(k);
        for (int i = 0; i < cols; ++i) {
            if (*(data + i) > 0) {
                sigma += ((*(data + i) - mean)*(*(data + i) - mean));
            }
        }
    }
    sigma /= nNum;
    sigma = sqrt(sigma);
    if (mean + 2 * sigma > 255) {
        sigma = (255 - mean) / 2.0;
    }
    float p = log(m0 / (m0 + 2 * sigma0)) / log(mean / (mean + 2 * sigma));
    float alpha = m0 / pow(mean, p);
    for (k = 0; k < rows; k++) {
        unsigned char* data = pData.ptr<uchar>(k);
        for (int i = 0; i < cols; ++i) {
            if (*(data + i) > 0) {
                float value = *(data + i);
                *(data + i) = alpha * pow(value, p);
            }
        }
    }
}

int ljocr::CharacterFeatureExtraction::extractFeaturebyHOG(const cv::Mat& character,const int background, vector<float>& feature) {
    Mat normalized_gray_img;
    Mat grad_x, grad_y;
    float m0 = 180;
    float sigma0 = 30;
    //normalizeGrayscale(character, m0, sigma0, background, normalized_gray_img);
    //cv::Sobel(normalized_gray_img, grad_x, CV_32F, 1, 0, 3);
    //cv::Sobel(normalized_gray_img, grad_y, CV_32F, 0, 1, 3);
    cv::Sobel(character, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(character, grad_y, CV_32F, 0, 1, 3);
    vector<Mat> direction_planes;
    for (int i = 0; i < num_direction_; ++i) {
        Mat direction_plane(cv::Size(character.cols, character.rows), CV_32F, cv::Scalar(0));
        direction_planes.push_back(direction_plane);
    }

    float bins_per_rad = 2.0 * CV_PI / num_direction_;
    for (int h = 0; h < character.rows; ++h) {
        for (int w = 0; w < character.cols; ++w) {
            float dx = grad_x.at<float>(h, w);
            float dy = grad_y.at<float>(h, w);
            float magtitude = sqrt(dx*dx + dy*dy);
            float orientation = atan2(dy, dx);
            if (orientation < 0) orientation += 2.0 * CV_PI;
            int orientation_bin = orientation / bins_per_rad;
            double alpha = (orientation_bin + 1)*bins_per_rad - orientation;
            double beta = orientation - orientation_bin*bins_per_rad;
            if (beta < 0) beta = 0;

            double bo = magtitude / sin(CV_PI - alpha - beta) * sin(alpha);  //sin theorem
            double eo = magtitude / sin(CV_PI - alpha - beta) * sin(beta);  //sin theorem
#if _DEBUG
            //if (bo < 0 || eo < 0)
            //{
            //  printf("debug\n");
            //}
#endif
            direction_planes[orientation_bin].at<float>(h, w) += bo;
            direction_planes[(orientation_bin + 1) % num_direction_].at<float>(h, w) += eo;
        }
    }
#if _DEBUG
    //for (vector<Mat>::iterator iter = direction_planes.begin(); iter != direction_planes.end(); ++iter)
    //{
    //  Mat temp = *iter;
    //  printf("debug\n");
    //}
#endif
    int pixels_per_block = character.cols / num_block_;
    for (vector<Mat>::iterator iter = direction_planes.begin(); iter != direction_planes.end(); ++iter) {
        Mat& direction_plane = *iter;
        double sigma = sqrt(2.0)*pixels_per_block / CV_PI;
        //lj blurring and sampling
        double sigma2 = sigma*sigma;
        //double reciprocal = 1.0/sqrt(2.0*CV_PI*sigma2);
        double reciprocal = 1.0/(2.0*CV_PI*sigma2);
        vector<float> fea;
        for (int j = 0; j < num_block_; ++j) {
            for (int i = 0; i < num_block_; ++i) {
                float begin_x = i*pixels_per_block;
                float end_x = (i+1)*pixels_per_block;
                float begin_y = j*pixels_per_block;
                float end_y = (j+1)*pixels_per_block;
                float center_x = (begin_x + end_x - 1)/2.0;
                float center_y = (begin_y + end_y - 1)/2.0;
                float sample_value = 0;
                for (int y = begin_y - pixels_per_block*0.5; y < end_y + pixels_per_block*0.5; ++y) {
                    if (y < 0 || y >= direction_plane.rows) {
                        continue;
                    }
                    float* ptr = direction_plane.ptr<float>(y);
                    for (int x = begin_x - pixels_per_block*0.5; x < end_x + pixels_per_block*0.5; ++x) {
                        if (x < 0 || x >= direction_plane.cols) {
                            continue;
                        }
                        float value = ptr[x];
                        sample_value += value * (exp(-1.0*((x-center_x)*(x-center_x)+(y-center_y)*(y-center_y))/(2*sigma2)) * reciprocal);
                    }
                }
                //fea.push_back(sqrt(sample_value)); //Box-cox
                fea.push_back((int)(sample_value+0.5)); 
            }
        }
        //lj end
        feature.insert(feature.end(), fea.begin(), fea.end());
    }
    return 0;
}

}