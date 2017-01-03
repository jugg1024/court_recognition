#include "FeatureStorage.h"

//#include <io.h>
#include <stdio.h>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace ljocr {

using cv::Mat;
using std::vector;

FeatureStorage::FeatureStorage(int mode) :mode_(mode) {}

FeatureStorage::~FeatureStorage() {}

int FeatureStorage::save(const std::string& filename, std::vector<std::pair<std::string, cv::Mat>>& feature) {
    if (feature.empty()) {
        printf("no feature\n");
        return -1;
    }
    if (mode_ == 0) {
        bool exist_flag = false;
        cv::FileStorage fs(filename, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            exist_flag = true;
            printf("Feature-File exists\n");
            fs.release();
        }
        if (exist_flag) {
            fs.open(filename, cv::FileStorage::APPEND);
        } else {
            fs.open(filename, cv::FileStorage::WRITE);
        }
        if (!fs.isOpened()) {
            printf("Feature-File cannot be opened\n");
            return -1;
        }
        for (std::vector<std::pair<std::string, Mat>>::iterator iter = feature.begin(); iter != feature.end(); ++iter) {
            fs << "character" << iter->first;
            fs << "feature" << iter->second;
        }
        fs.release();
    } else if (mode_ == 1) {
        FILE* file = fopen(filename.c_str(), "ab+");
        if (file == NULL) {
            printf("Feature-File cannot be opened\n");
            return -1;
        }
        for (std::vector<std::pair<std::string, Mat>>::iterator iter = feature.begin(); iter != feature.end(); ++iter) {
            std::string name = iter->first;
            name += "\n";
            const char* character = name.c_str();
            fwrite(character, sizeof(char), strlen(character), file);

            Mat mtx(iter->second);
            mtx.convertTo(mtx, CV_32F);
            fwrite(&mtx.rows, sizeof(mtx.rows), 1, file);
            fwrite(&mtx.cols, sizeof(mtx.cols), 1, file);
            for (int j = 0; j < mtx.rows; ++j) {
                float* ptr = mtx.ptr<float>(j);
                fwrite(ptr, sizeof(float), mtx.cols, file);
            }
        }
        feature.clear();
        fclose(file);
    } else {
        printf("error mode\n");
    }
    return 0;
}


int FeatureStorage::load(const std::string& filename, std::vector<std::pair<std::string, cv::Mat>>& feature) {
    if (mode_ == 0) {
        cv::FileStorage fs(filename, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            printf("Feature-File cannot be opened\n");
            return -1;
        }
        for (std::vector<std::pair<std::string, Mat>>::iterator iter = feature.begin(); iter != feature.end(); ++iter) {
            fs["character"] >> iter->first;
            fs["feature"] >> iter->second;
        }
        fs.release();
    } else if (mode_ == 1) {
        FILE* file = fopen(filename.c_str(), "rb");
        if (file == NULL) {
            printf("Feature-File cannot be opened\n");
            return -1;
        }
        feature.clear();
        char character[1024] = {'\0'};
        while (fgets(character, 1024, file)) {
            character[strlen(character) - 1] = {'\0'};
            int rows = 0;
            int cols = 0;
            fread(&rows, sizeof(rows), 1, file);
            fread(&cols, sizeof(cols), 1, file);
            Mat mtx(cv::Size(cols, rows), CV_32F, cv::Scalar(0));
            for (int j = 0; j < mtx.rows; ++j) {
                float* ptr = mtx.ptr<float>(j);
                fread(ptr, sizeof(float), mtx.cols, file);
            }
            feature.push_back(std::make_pair(character, mtx));
        }
        fclose(file);
    } else {
        printf("error mode\n");
    }
    return 0;
}

}