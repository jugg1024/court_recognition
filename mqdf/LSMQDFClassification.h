#ifndef LSMQDFCLASSFICATION_H_
#define LSMQDFCLASSFICATION_H_

#include <vector>
#include <map>
#include <string>
#include <functional>
#include "opencv2/highgui/highgui.hpp"

#include "MQDFParam.h"

namespace ljocr {


class LSMQDFClassification {
  public:
    LSMQDFClassification();
    ~LSMQDFClassification();
    int addSamples(std::string character, std::vector<std::vector<float> >& samples);
    int train(float ratio, std::string path);

	//recognize
    int loadTemplate(std::string path);
    int recognize(std::vector<float>& feature, int num_candidate, std::map<double, std::string, std::greater<double> >& recognition_result);

    std::map<std::string, MQDFParam> character_mqdf_param_;
  protected:
  private:
};
}

#endif