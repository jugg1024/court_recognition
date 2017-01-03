#include "CharacterNormalization.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <math.h>


namespace ljocr {

using cv::Mat;
CharacterNormalization::CharacterNormalization(int normalized_size) {
    normalized_size_ = normalized_size;
}

CharacterNormalization::~CharacterNormalization() {

}

static void BiMoments(Mat& pData, float& centroid_x, float& centroid_y, float& negtive_u_x, float& plus_u_x, float& negtive_u_y, float& plus_u_y) {
    int j, k;
    int rows = pData.rows;
    int cols = pData.cols;
    unsigned char *pTr = pData.data;

    float *pHx = new float[cols];
    float *pHy = new float[rows];
    float total = 0;

    memset(pHx, 0, cols * sizeof(float));
    memset(pHy, 0, rows * sizeof(float));

    for (k = 0; k < rows; k++) {
        for (j = 0; j < cols; j++) {
            if (*(pTr + k * cols + j) > 0) {
                total += *(pTr + k * cols + j);
                *(pHx + j) += *(pTr + k * cols + j);
                *(pHy + k) += *(pTr + k * cols + j);
            }//
        }//
    }
    //
    float xc = 0;
    float yc = 0;
    // centroid (xc, yc)
    for (j = 0; j < cols; j++) {
        *(pHx + j) /= total;
        xc += *(pHx + j) * (j + (float)0.01);
    }
    //
    for (k = 0; k < rows; k++) {
        *(pHy + k) /= total;
        yc += *(pHy + k) * (k + (float)0.01);
    }
    // one-sided second-order moments
    //
    float rn, rp;

    float px = 0;
    float nx = 0;
    rn = rp = 0;
    for (j = 0; j < cols; j++) {
        if (j < xc) {
            nx += *(pHx + j) * (j + (float)0.01 - xc) * (j + (float)0.01 - xc);
            rn += *(pHx + j);
        } else {
            px += *(pHx + j) * (j + (float)0.01 - xc) * (j + (float)0.01 - xc);
            rp += *(pHx + j);
        }
    }
    //
    if (rn > 0) nx = sqrt(nx / rn);
    if (rp > 0) px = sqrt(px / rp);

    //
    float py = 0;
    float ny = 0;
    rn = rp = 0;
    for (k = 0; k < rows; k++) {
        if (k < yc) {
            ny += *(pHy + k) * (k + (float)0.01 - yc) * (k + (float)0.01 - yc);
            rn += *(pHy + k);
        } else {
            py += *(pHy + k) * (k + (float)0.01 - yc) * (k + (float)0.01 - yc);
            rp += *(pHy + k);
        }
    }
    if (rn > 0) ny = sqrt(ny / rn);
    if (rp > 0) py = sqrt(py / rp);

    delete[]pHx;
    delete[]pHy;
    centroid_x = xc;
    centroid_y = yc;
    negtive_u_x = nx;
    negtive_u_y = ny;
    plus_u_x = px;
    plus_u_y = py;
}

static float QuadraticFunction(float fa, float fb, float fc, float x) {
    return (fa * x * x + fb * x + fc);
}

static void QuadraticFunctionFitting(float begin_x, float end_x, float centroid_x, float begin_y, float end_y, float centroid_y, float& a_x, float& b_x, float& c_x, float& a_y, float& b_y, float& c_y) {
    a_x = (1.0 / (end_x - begin_x) - 0.5 / (centroid_x - begin_x)) / (end_x - centroid_x);
    b_x = 0.5 / (centroid_x - begin_x) - a_x*(centroid_x + begin_x);
    //b_x = 1.0 / (end_x - begin_x) - a_x*(end_x + begin_x);
    c_x = 0 - a_x*begin_x*begin_x - b_x*begin_x;
    //c_x = 1.0 - a_x*end_x*end_x - b_x*end_x;
    a_y = (1.0 / (end_y - begin_y) - 0.5 / (centroid_y - begin_y)) / (end_y - centroid_y);
    b_y = 0.5 / (centroid_y - begin_y) - a_y*(centroid_y + begin_y);
    c_y = 0 - a_y*begin_y*begin_y - b_y*begin_y;
}

static void ForwardMapping(float begin_y, float end_y, float begin_x, float end_x, cv::Mat &src_character, int normalized_size, float a_x, float b_x, float c_x, float a_y, float b_y, float c_y, Mat &normal_character_) {
    for (int y = begin_y; y <= end_y; y++) {
        for (int x = begin_x; x <= end_x; x++) {
            unsigned char src_pixel = src_character.at<uchar>(y, x);
            if (src_pixel == 0) continue;
            float normal_begin_x = normalized_size * QuadraticFunction(a_x, b_x, c_x, x);
            float normal_end_x = normalized_size * QuadraticFunction(a_x, b_x, c_x, x + 1);
            float normal_begin_y = normalized_size * QuadraticFunction(a_y, b_y, c_y, y);
            float normal_end_y = normalized_size * QuadraticFunction(a_y, b_y, c_y, y + 1);
            for (int ny = normal_begin_y; ny <= normal_end_y; ny++) {
                if (ny < 0 || ny >= normalized_size) continue;
                for (int nx = normal_begin_x; nx <= normal_end_x; nx++) {
                    if (nx < 0 || nx >= normalized_size) continue;
                    float overlap_bx = MAX(normal_begin_x, nx);
                    float overlap_ex = MIN(normal_end_x, nx + 1);
                    float overlap_by = MAX(normal_begin_y, ny);
                    float overlap_ey = MIN(normal_end_y, ny + 1);
                    int value = normal_character_.at<uchar>(ny, nx);
                    value += (overlap_ex - overlap_bx)*(overlap_ey - overlap_by)*src_pixel;
                    if (value > 255) {
                        value = 255;
                    }
                    normal_character_.at<uchar>(ny, nx) = value;
                }
            }
        }
    }
}

int CharacterNormalization::normalizebyBiMoment(cv::Mat& src_character, cv::Mat& normal_character) {
    if (src_character.channels() != 1) return -1;
    //float m0 = 180;
    //float sigma0 = 30;
    //Mat character = src_character.clone();
    //normalizeGrayscale(character, m0, sigma0, src_character);
    float centroid_x, centroid_y, negtive_u_x, negtive_u_y, plus_u_x, plus_u_y;
    BiMoments(src_character, centroid_x, centroid_y, negtive_u_x, plus_u_x, negtive_u_y, plus_u_y);
    float begin_x = 0;
    if (centroid_x - 2 * negtive_u_x > 0)
        begin_x = centroid_x - 2 * negtive_u_x;
    float begin_y = 0;
    if (centroid_y - 2 * negtive_u_y > 0)
        begin_y = centroid_y - 2 * negtive_u_y;
    float end_x = centroid_x + 2 * plus_u_x;
    if (end_x > src_character.cols)
        end_x = src_character.cols - 1;
    float end_y = centroid_y + 2 * plus_u_y;
    if (end_y > src_character.rows)
        end_y = src_character.rows - 1;

    float a_x, b_x, c_x;
    float a_y, b_y, c_y;
    QuadraticFunctionFitting(begin_x, end_x, centroid_x, begin_y, end_y, centroid_y, a_x, b_x, c_x, a_y, b_y, c_y);

    Mat normal_character_temp(cv::Size(normalized_size_, normalized_size_), CV_8U, cv::Scalar(0));//建立一个mat，初值为0
    ForwardMapping(begin_y, end_y, begin_x, end_x, src_character, normalized_size_, a_x, b_x, c_x, a_y, b_y, c_y, normal_character_temp);
    normal_character = normal_character_temp;
    return 0;
}

//Handwritten digit recognition: investigation of normalization and feature extraction techniques
//method F8 and F9
int CharacterNormalization::normalizebyARAN(cv::Mat& src_character, cv::Mat& normal_character) {
    if (src_character.channels() != 1) return -1;
    //float m0 = 180;
    //float sigma0 = 30;
    //Mat character = src_character.clone();
    //normalizeGrayscale(character, m0, sigma0, src_character);
    float centroid_x, centroid_y, negtive_u_x, negtive_u_y, plus_u_x, plus_u_y;
    cv::Moments moments = cv::moments(src_character);
    centroid_x = moments.m10 / moments.m00;
    centroid_y = moments.m01 / moments.m00;

    float W1 = 4 * sqrt(moments.mu20 / moments.m00);
    float H1 = 4 * sqrt(moments.mu02 / moments.m00);
    float R1 = MIN(W1, H1) / (MAX(W1, H1));
    float R2 = cbrt(R1);
    float H2;
    float W2;
    if (H1 > W1) {
        H2 = normalized_size_;
        W2 = R2 * H2;
    } else {
        W2 = normalized_size_;
        H2 = R2 * W2;
    }

    float x_scale = W2 / W1; //alpha
    float y_scale = H2 / H1; //beta

    float normalized_centroid_x = W2 / 2;
    float normalized_centroid_y = H2 / 2;

    Mat normal_character_temp(cv::Size(normalized_size_, normalized_size_), CV_8U, cv::Scalar(0));//建立一个mat，初值为0
    float begin_y = centroid_y - H1 / 2;
    if (begin_y < 0) begin_y = 0;
    float end_y = centroid_y + H1 / 2;
    if (end_y > src_character.rows) end_y = H1 - 1;
    if (end_y > src_character.rows) end_y = src_character.rows - 1;
    float begin_x = centroid_x - W1 / 2;
    if (begin_x < 0) begin_x = 0;
    float end_x = centroid_x + W1 / 2;
    if (end_x > src_character.cols) end_x = W1 - 1;
    if (end_x > src_character.cols) end_x = src_character.cols - 1;
    for (int y = begin_y; y <= end_y; y++) {
        for (int x = begin_x; x <= end_x; x++) {
            unsigned char src_pixel = src_character.at<uchar>(y, x);
            if (src_pixel == 0) continue;
            float normal_begin_x = x_scale * (x - centroid_x) + normalized_centroid_x;
            float normal_end_x = x_scale * (x + 1 - centroid_x) + normalized_centroid_x;
            float normal_begin_y = y_scale * (y - centroid_y) + normalized_centroid_y;
            float normal_end_y = y_scale * (y + 1 - centroid_y) + normalized_centroid_y;
            for (int ny = normal_begin_y; ny <= normal_end_y; ny++) {
                if (ny < 0 || ny >= normalized_size_) continue;
                for (int nx = normal_begin_x; nx <= normal_end_x; nx++) {
                    if (nx < 0 || nx >= normalized_size_) continue;
                    float overlap_bx = MAX(normal_begin_x, nx);
                    float overlap_ex = MIN(normal_end_x, nx + 1);
                    float overlap_by = MAX(normal_begin_y, ny);
                    float overlap_ey = MIN(normal_end_y, ny + 1);
                    int value = normal_character_temp.at<uchar>(ny, nx);
                    value += (overlap_ex - overlap_bx)*(overlap_ey - overlap_by)*src_pixel;
                    if (value > 255) {
                        value = 255;
                    }
                    normal_character_temp.at<uchar>(ny, nx) = value;
                }
            }
        }
    }
    normal_character = normal_character_temp;
    return 0;
}

//Online and Offline Handwritten Chinese Character Recognition: Benchmarking on New Databases
//Standardize the gray scale level
//float m0 = 180; float sigma0 = 30;
void CharacterNormalization::normalizeGrayscale(Mat& src, float m0, float sigma0, Mat& pData) {
    pData = src.clone();
    int cols = pData.cols;
    int rows = pData.rows;
    if (pData.isContinuous()) {
        cols = pData.cols * pData.rows;
        rows = 1;
    }
    //eliminate background noise using OTSU
    cv::threshold(src, pData, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    for (int k = 0; k < rows; k++) {
        unsigned char* data = pData.ptr<uchar>(k);
        unsigned char* src_data = src.ptr<uchar>(k);
        for (int i = 0; i < cols; ++i) {
            //if (*(data + i) == 0)  //when the background is white
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
            *(data + i) = 255 - *(data + i); //reverse image when the background is white
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
    for (k = 0; k < rows; k++) {
        unsigned char* data = pData.ptr<uchar>(k);
        for (int i = 0; i < cols; ++i) {
            float value = *(data + i);
            *(data + i) = 255 - alpha * pow(value, p);
        }
    }
}

}