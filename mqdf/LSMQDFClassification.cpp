#include "LSMQDFClassification.h"
//#include <io.h>
#include <stdio.h>
#include <functional>

#include <map>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

//#include "MQDFClassification.h"
#include "LDADimensionReduction.h"
#include "FeatureStorage.h"
#include "CommonTools.h"

#define TRAIN

using std::vector;
using std::string;
using namespace cv;

namespace ljocr {


static int writeTemplate(std::string character, MQDFParam& param, std::string path) {
    FILE* tlp = fopen(path.c_str(), "rb");
    if (tlp != NULL) {
        fclose(tlp);
        tlp = fopen(path.c_str(), "ab+");
        fseek(tlp, 0, SEEK_SET);
        char header[5] = { '\0' };
        fread(header, sizeof(char), 4, tlp);
        if (strcmp(header, "MQDF") == 0) {
            fseek(tlp, 0, SEEK_END);
        } else {
            fclose(tlp);
            return -1;
        }
    } else {
        tlp = fopen(path.c_str(), "wb+");
        fwrite("MQDF", sizeof(char), 4, tlp);
    }
    if (tlp == NULL) {
        return -1;
    }
    character += "\n";
    const char* cha = character.c_str();
    fwrite(cha, sizeof(char), character.length(), tlp);
    fwrite(&param.remainder, sizeof(param.remainder), 1, tlp);
    fwrite(&param.delta, sizeof(param.delta), 1, tlp);
    fwrite(&param.principle_dim, sizeof(param.principle_dim), 1, tlp);
    fwrite(&param.principle_eigenvalue.rows, sizeof(param.principle_eigenvalue.rows), 1, tlp);
    fwrite(&param.principle_eigenvalue.cols, sizeof(param.principle_eigenvalue.cols), 1, tlp);
    for (int j = 0; j < param.principle_eigenvalue.rows; ++j) {
        float* ptr = param.principle_eigenvalue.ptr<float>(j);
        fwrite(ptr, sizeof(float), param.principle_eigenvalue.cols, tlp);
    }

    fwrite(&param.principle_eigenvector.rows, sizeof(param.principle_eigenvector.rows), 1, tlp);
    fwrite(&param.principle_eigenvector.cols, sizeof(param.principle_eigenvector.cols), 1, tlp);
    for (int j = 0; j < param.principle_eigenvector.rows; ++j) {
        float* ptr = param.principle_eigenvector.ptr<float>(j);
        fwrite(ptr, sizeof(float), param.principle_eigenvector.cols, tlp);
    }

    fwrite(&param.mean.rows, sizeof(param.mean.rows), 1, tlp);
    fwrite(&param.mean.cols, sizeof(param.mean.cols), 1, tlp);
    for (int j = 0; j < param.mean.rows; ++j) {
        float* ptr = param.mean.ptr<float>(j);
        fwrite(ptr, sizeof(float), param.mean.cols, tlp);
    }
    fclose(tlp);
    return 0;
}

LSMQDFClassification::LSMQDFClassification() {

}

LSMQDFClassification::~LSMQDFClassification() {

}

int LSMQDFClassification::addSamples(std::string character, std::vector<std::vector<float> >& samples) {
    int label = 0;
    cv::Mat mtx_samples = Vect2MatEx(samples);
    cv::Mat cov, mean;
    calcCovarMatrix(mtx_samples, cov, mean, CV_COVAR_NORMAL | CV_COVAR_ROWS);
    MQDFParam mqdf_param;
    mean.convertTo(mean, CV_32F);
    mqdf_param.mean = mean.clone();
    cov /= (mtx_samples.rows - 1);   //cov/(N-1)
    mqdf_param.cov = cov.clone();
    mqdf_param.cov.convertTo(mqdf_param.cov, CV_32F);
    character_mqdf_param_.insert(std::make_pair(character, mqdf_param));
    return 0;
}

int LSMQDFClassification::train(float ratio, std::string template_path) {
    std::map<std::string, Mat> character_cov;
    for (std::map<std::string, MQDFParam>::iterator iter = character_mqdf_param_.begin(); iter != character_mqdf_param_.end(); ++iter) {
        Mat x = iter->second.mean.clone();
        Mat origin_cov = iter->second.cov;
        std::map<double, string, std::less<double> > candidate;
        for (std::map<string, MQDFParam>::iterator iter2 = character_mqdf_param_.begin(); iter2 != character_mqdf_param_.end(); ++iter2) {
            Mat mean = iter2->second.mean;
            Mat x_u = x - mean;
            x_u.convertTo(x_u, CV_64F);
            Mat euclidean_dis = x_u*x_u.t();
            if (euclidean_dis.at<double>(0, 0) == 0) {
                continue;
            }
            int euclidean_num_candidate = 10;
            if (candidate.size() < euclidean_num_candidate) {
                candidate.insert(make_pair(euclidean_dis.at<double>(0, 0), iter2->first));
            } else {
                std::map<double, string, std::less<double> >::reverse_iterator candidat_iter = candidate.rbegin();
                if (candidat_iter->first > euclidean_dis.at<double>(0, 0)) {
                    candidate.erase(--candidate.end());
                    candidate.insert(make_pair(euclidean_dis.at<double>(0, 0), iter2->first));
                }
            }
        }
        //lj debug
        double weight_sum = 0;
        for (std::map<double, string, std::less<double> >::iterator candidate_iter = candidate.begin(); candidate_iter != candidate.end(); ++candidate_iter) {
            weight_sum += candidate_iter->first;
        }
        weight_sum = candidate.rbegin()->first;
        //lj end
        std::map<double, string, std::less<double> >::iterator candidate_iter = candidate.begin();
        Mat neighbers_cov_sum = character_mqdf_param_[candidate_iter->second].cov.clone();
        //lj debug
        double weight = candidate_iter->first / weight_sum;
        weight = 1 - weight;
        neighbers_cov_sum *= weight;
        printf("weight = %f\n", weight);
        //lj end
        ++candidate_iter;
        for (; candidate_iter != candidate.end(); ++candidate_iter) {
            //neighbers_cov_sum += character_mqdf_param_[candidate_iter->second].cov;
            //lj debug
            weight = candidate_iter->first / weight_sum;
            weight = 1 - weight;
            printf("weight = %f\n", weight);
            neighbers_cov_sum += weight * character_mqdf_param_[candidate_iter->second].cov;
            //lj end
        }
        Mat new_cov = 0.8 * iter->second.cov;
        new_cov += 0.2 * neighbers_cov_sum;
        //new_cov += 0.05 * neighbers_cov_sum;
        //Mat new_w, new_u, new_vt;
        //svd.compute(new_cov, new_w, new_u, new_vt, SVD::FULL_UV);
        character_cov.insert(std::make_pair(iter->first, new_cov));
    }
    for (std::map<std::string, MQDFParam>::iterator iter = character_mqdf_param_.begin(); iter != character_mqdf_param_.end(); ++iter) {
        iter->second.cov = character_cov[iter->first];
    }

    int i = 0;
    for (std::map<std::string, MQDFParam>::iterator iter = character_mqdf_param_.begin(); iter != character_mqdf_param_.end(); ++iter) {
        MQDFParam mqdf_param;
        Mat cov = iter->second.cov;
        //cov /= (samples.rows - 1);   //cov/(N-1)
        SVD svd;
        Mat w, u, vt;
        svd.compute(cov, w, u, vt, SVD::FULL_UV);
        double sum_w = 0;
        for (int j = 0; j < w.rows; ++j) {
            sum_w += w.at<float>(j, 0);
        }
        mqdf_param.principle_dim = 0;
        double log_lamda = 0;
        if (ratio < 1.0) {
            int j;
            double temp_sum = 0;
            for (j = 0; j < w.rows; ++j) {
                double lamda_temp = w.at<float>(j, 0);
                temp_sum += w.at<float>(j, 0);
                if (temp_sum / sum_w > ratio) {
                    break;
                }
                log_lamda += log(lamda_temp);
            }
            mqdf_param.principle_dim = j;
        } else {
            mqdf_param.principle_dim = ratio;
            for (int j = 0; j < mqdf_param.principle_dim; ++j) {
                double lamda_temp = w.at<float>(j, 0);
                log_lamda += log(lamda_temp);
            }
        }
        float principle_eigenvalue_sum = 0;
        for (int j = 0; j < mqdf_param.principle_dim; ++j) {
            principle_eigenvalue_sum += w.at<float>(j, 0);
        }
#ifdef TRAIN
        mqdf_param.delta = sum_w / w.rows;  //prepare for cross validation
#else
        //mqdf_param.delta = 27.6436;  //the value is estimated by cross_validation for 300 samples
        //mqdf_param.delta = 26.2426969;  //the value is estimated by cross_validation for 1000 samples
        //mqdf_param.delta = 32.6721689;  //the value is estimated by cross_validation for 3755 samples
        //mqdf_param.delta = 64.919496*0.7;  //the value is estimated by cross_validation for 3755 samples
        //mqdf_param.delta = 1.896899*0.7;  //the value is estimated by cross_validation for 3755 samples. best for MPF
        //mqdf_param.delta = 1.361794*0.7;  //the value is estimated by cross_validation for 500 samples. best for MPF
        //mqdf_param.delta = 23.766482*0.7;  //the value is estimated by cross_validation for 3755 samples
        //mqdf_param.delta = 0.960328*0.7;  //the value is estimated by cross_validation for 500 samples. best for MPF class 0
        //mqdf_param.delta = 1.114572*0.7;  //the value is estimated by cross_validation for 500 samples. best for MPF class 1
        //mqdf_param.delta = 0.997763*0.7;  //the value is estimated by cross_validation for 500 samples. best for MPF class 1
        //mqdf_param.delta = 1.604470*0.7;  //the value is estimated by cross_validation for 1000 samples. best for MPF
        //mqdf_param.delta = 1.339773*0.7;  //the value is estimated by cross_validation for 1000 samples. best for MPF class 0
        //mqdf_param.delta = 1.228499*0.7;  //the value is estimated by cross_validation for 1000 samples. best for MPF class 1
        //mqdf_param.delta = 1.677936*0.7;  //the value is estimated by cross_validation for 3755 samples. best for MPF class 0
        //mqdf_param.delta = 1.675047*0.7;  //the value is estimated by cross_validation for 3755 samples. best for MPF class 1
        //mqdf_param.delta = 1.541195*0.7;  //the value is estimated by cross_validation for 3755 samples. best for MPF class 2, 3 class
        //mqdf_param.delta = 1.571957*0.7;  //the value is estimated by cross_validation for 3755 samples. best for MPF class 1, 3 class
        //mqdf_param.delta = 1.52195*0.7;  //the value is estimated by cross_validation for 3755 samples. best for MPF class 0, 3 class

        //mqdf_param.delta = 1.385178*0.7;  //the value is estimated by cross_validation for 3755 samples. best for MPF class 4, 5 class
        //mqdf_param.delta = 1.430945*0.7;  //the value is estimated by cross_validation for 3755 samples. best for MPF class 3, 5 class
        //mqdf_param.delta = 1.249550*0.7;  //the value is estimated by cross_validation for 3755 samples. best for MPF class 2, 5 class
        //mqdf_param.delta = 1.322204*0.7;  //the value is estimated by cross_validation for 3755 samples. best for MPF class 1, 5 class
        //mqdf_param.delta = 1.358019*0.7;  //the value is estimated by cross_validation for 3755 samples. best for MPF class 0, 5 class

        //mqdf_param.delta = 1.496428*0.7;  //the value is estimated by cross_validation for 3755 samples. best for MPF class 0, 5 class pca95
        //mqdf_param.delta = 1.435780*0.7;  //the value is estimated by cross_validation for 3755 samples. best for MPF class 1, 5 class pca95
        //mqdf_param.delta = 1.333554*0.7;  //the value is estimated by cross_validation for 3755 samples. best for MPF class 2, 5 class pca95
        //mqdf_param.delta = 1.287792*0.7;  //the value is estimated by cross_validation for 3755 samples. best for MPF class 3, 5 class pca95
        //mqdf_param.delta = 1.309553*0.7;  //the value is estimated by cross_validation for 3755 samples. best for MPF class 4, 5 class pca95
        //mqdf_param.delta = 1.904013*0.7;  //the value is estimated by cross_validation for 3755 samples. best for MPF class 4, 5 class pca95
        //mqdf_param.delta = 1.845922*0.7;  //the value is estimated by cross_validation for 300 samples.
        //mqdf_param.delta = 259.5905*0.7;  //the value is estimated by cross_validation for 300 samples.
        //mqdf_param.delta = 3.482881*0.7;  //the value is estimated by cross_validation for 300 samples.
        //mqdf_param.delta = 145.980744*0.7;  //the value is estimated by cross_validation for 300 samples.
        //mqdf_param.delta = 295.230675*0.7;  //the value is estimated by cross_validation for 300 samples.
        //mqdf_param.delta = 2.279374*0.7;  //the value is estimated by cross_validation for 300 samples.
        mqdf_param.delta = 1.524346*0.7;  //the value is estimated by cross_validation for 800 samples.
#endif
        mqdf_param.remainder = log_lamda;
        //lj end
        Rect rect(0, 0, 1, mqdf_param.principle_dim);
        cv::Mat principle_eigenvalue = w(rect).clone();
        principle_eigenvalue.convertTo(principle_eigenvalue, CV_32F);
        mqdf_param.principle_eigenvalue = principle_eigenvalue;
        rect.x = 0;
        rect.y = 0;
        rect.height = u.rows;
        rect.width = mqdf_param.principle_dim;
        cv::Mat principle_eigenvector = u(rect).clone();
        principle_eigenvector.convertTo(principle_eigenvector, CV_32F);
        mqdf_param.principle_eigenvector = principle_eigenvector;
#ifdef TRAIN
        //character_mqdf_param_.insert(std::make_pair(character, mqdf_param));
        iter->second.remainder = mqdf_param.remainder;
        iter->second.delta = mqdf_param.delta;
        //iter->second.mean = mqdf_param.mean;
        iter->second.principle_dim = mqdf_param.principle_dim;
        iter->second.principle_eigenvalue = mqdf_param.principle_eigenvalue;
        iter->second.principle_eigenvector = mqdf_param.principle_eigenvector;
#else
        //writeTemplate(character, mqdf_param, path);
#endif
        printf("i = %d\n", i++);
    }
    //compute test delta
    double delta = 0;
    for (std::map<std::string, MQDFParam>::iterator iter = character_mqdf_param_.begin(); iter != character_mqdf_param_.end(); ++iter) {
        delta += iter->second.delta;
    }
    delta /= character_mqdf_param_.size();
    printf("delta = %f\n", delta);
    for (std::map<std::string, MQDFParam>::iterator iter = character_mqdf_param_.begin(); iter != character_mqdf_param_.end(); ++iter) {
        iter->second.delta = delta*0.7;
        writeTemplate(iter->first, iter->second, template_path);
    }
    return 0;
}

int LSMQDFClassification::loadTemplate(std::string path) {
    FILE* tlp = fopen(path.c_str(), "rb");
    if (tlp == NULL) {
        return -1;
    }
    int byte_len = 0;
    char header[5] = { '\0' };
    fread(header, sizeof(char), 4, tlp);
    if (strcmp(header, "MQDF") != 0) {
        return -2;
    }
    character_mqdf_param_.clear();
    char character[1024] = { '\0' };
    while (fgets(character, 1024, tlp)) {
        character[strlen(character) - 1] = { '\0' };
        MQDFParam param;
        fread(&param.remainder, sizeof(param.remainder), 1, tlp);
        fread(&param.delta, sizeof(param.delta), 1, tlp);
        fread(&param.principle_dim, sizeof(param.principle_dim), 1, tlp);
        int rows = 0;
        int cols = 0;
        fread(&rows, sizeof(rows), 1, tlp);
        fread(&cols, sizeof(cols), 1, tlp);
        Mat principle_eigenvalue(Size(cols, rows), CV_32F, cv::Scalar(0));
        for (int j = 0; j < principle_eigenvalue.rows; ++j) {
            float* ptr = principle_eigenvalue.ptr<float>(j);
            fread(ptr, sizeof(float), principle_eigenvalue.cols, tlp);
        }
        principle_eigenvalue.convertTo(principle_eigenvalue, CV_64F);
        param.principle_eigenvalue = principle_eigenvalue;

        rows = 0;
        cols = 0;
        fread(&rows, sizeof(rows), 1, tlp);
        fread(&cols, sizeof(cols), 1, tlp);
        Mat principle_eigenvector(Size(cols, rows), CV_32F, cv::Scalar(0));
        for (int j = 0; j < principle_eigenvector.rows; ++j) {
            float* ptr = principle_eigenvector.ptr<float>(j);
            fread(ptr, sizeof(float), principle_eigenvector.cols, tlp);
        }
        principle_eigenvector.convertTo(principle_eigenvector, CV_64F);
        param.principle_eigenvector = principle_eigenvector;

        fread(&rows, sizeof(rows), 1, tlp);
        fread(&cols, sizeof(cols), 1, tlp);
        Mat mean(Size(cols, rows), CV_32F, cv::Scalar(0));
        for (int j = 0; j < mean.rows; ++j) {
            float* ptr = mean.ptr<float>(j);
            fread(ptr, sizeof(float), mean.cols, tlp);
        }
        mean.convertTo(mean, CV_64F);
        param.mean = mean;
        character_mqdf_param_.insert(std::make_pair(character, param));
    }
    fclose(tlp);
    return 0;
}

int LSMQDFClassification::recognize(std::vector<float>& feature, int num_candidate, std::map<double, std::string, std::greater<double> >& recognition_result)
{
	Mat x(feature, true);
	x = x.t();
	x.convertTo(x, CV_64F);
	std::map<double, string, std::greater<double> > candidate;
	for (std::map<string, MQDFParam>::iterator iter = character_mqdf_param_.begin(); iter != character_mqdf_param_.end(); ++iter) {
		MQDFParam mqdf_param = iter->second;
		Mat mean;
		mqdf_param.mean.convertTo(mean, CV_64F);
		Mat x_u = x - mean;
		x_u.convertTo(x_u, CV_64F);
#ifdef EUCLIDEAN_ACCELERATION
		//Euclidean acceleration
		Mat euclidean_dis = x_u*x_u.t();
		int euclidean_num_candidate = 100;
		if (candidate.size() < euclidean_num_candidate) {
			candidate.insert(make_pair(euclidean_dis.at<double>(0, 0), iter->first));
		}
		else {
			std::map<double, string, std::greater<double> >::iterator candidat_iter = candidate.begin();
			if (candidat_iter->first > euclidean_dis.at<double>(0, 0)) {
				candidate.erase(candidat_iter);
				candidate.insert(make_pair(euclidean_dis.at<double>(0, 0), iter->first));
			}
		}
	}
	for (std::map<double, string, std::less<double> >::iterator candidat_iter = candidate.begin(); candidat_iter != candidate.end(); ++candidat_iter) {
		MQDFParam mqdf_param = character_mqdf_param_[candidat_iter->second];
		Mat mean;
		mqdf_param.mean.convertTo(mean, CV_64F);
		Mat x_u = x - mean;
		x_u.convertTo(x_u, CV_64F);
#endif
		Mat principle_eigenvalue;
		mqdf_param.principle_eigenvalue.convertTo(principle_eigenvalue, CV_64F);
		double sum = 0;
		double sum2 = 0;
		for (int dim = 0; dim < mqdf_param.principle_dim; ++dim) {
			Mat temp = mqdf_param.principle_eigenvector.col(dim);
			temp.convertTo(temp, CV_64F);
			temp = x_u * temp;
			double value = temp.at<double>(0, 0);
			value *= value;
			sum2 += value;
			double lamda = principle_eigenvalue.at<double>(dim, 0);
			value = value / lamda;
			sum += value;
		}
		x_u = x_u * x_u.t();
		double confidence = x_u.at<double>(0, 0);
		confidence -= sum2;
		confidence /= mqdf_param.delta;
		confidence += sum;
		confidence += mqdf_param.remainder;
		if (recognition_result.size() < num_candidate) {
#ifdef EUCLIDEAN_ACCELERATION
			recognition_result.insert(make_pair(confidence, candidat_iter->second));
#else
			recognition_result.insert(make_pair(confidence, iter->first));
#endif
		}
		else {
			std::map<double, string, std::greater<double> >::iterator result_iter = recognition_result.begin();
			if (result_iter->first > confidence) {
				recognition_result.erase(result_iter);
#ifdef EUCLIDEAN_ACCELERATION
				recognition_result.insert(make_pair(confidence, candidat_iter->second));
#else
				recognition_result.insert(make_pair(confidence, iter->first));
#endif
			}
		}
	}
	return recognition_result.rbegin()->first;
}

}


