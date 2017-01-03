#include "plate_recognizor.h"
#include "Bbox.h"

#include "chinese_traverse.h"
#include "LSMQDFClassification.h"

#include <iterator>
#include <numeric>
#include <vector>
#include <iostream>
#include <iconv.h> 
#define OUTLEN 255 

using namespace std;
using namespace cv;

const string dictionary[] = { "被告", "原告", "审判员", "审判长", "书记员",
"陪审员", "人民陪审员",
"辩护人", "上诉人", "被上诉人", "公诉人", "第三人",
"代理人", "委托代理人", "诉讼代理人", "法定代理人",
"代理审判员", "代理审判长",
"检察机关", "检察员", "被告人", "原告人",
"原审第三人", "原告代理人", "被告代理人", "证人", "鉴定人",
"承办人", "执行员", "法官助理", "审判委员会委员", "赔偿委员会委员", "司法警察", "技术调查官"};


int code_convert(char *from_charset, char *to_charset, char *inbuf, size_t inlen, char *outbuf, size_t outlen) { 
  iconv_t cd; 
  int rc; 
  char **pin = &inbuf; 
  char **pout = &outbuf; 
  cd = iconv_open(to_charset,from_charset); 
  if (cd==0) 
  	return -1; 
  memset(outbuf,0,outlen); 
  if (iconv(cd,pin,&inlen,pout,&outlen)==-1) 
  	return -1; 
  iconv_close(cd); 
  return 0; 
} 
 
int u2g(char *inbuf,int inlen,char *outbuf,int outlen) { 
  return code_convert("utf-8","gb2312",inbuf,inlen,outbuf,outlen); 
 } 
   
int g2u(char *inbuf,size_t inlen,char *outbuf,size_t outlen) { 
  return code_convert("gb2312","utf-8",inbuf,inlen,outbuf,outlen); 
} 

vector<string> split_words(string words)
{
	vector<string> words_chip;
	if (words == "")
	{
		words_chip.push_back(words);
		return words_chip;
	}

	int t = 0;
	char q[40];
	strcpy(q, words.c_str());


	string temp;
	for (int i = 0; i <words.length() - 2;)
	{
		temp = "";
		for (int j = 0; j < 4; j++)
		{
			temp += q[i + j];
		}
		words_chip.push_back(temp);

		i += 2;
	}

	return  words_chip;
}

int ldistance(const string source, const string target)
{
	//step 1  

	int n = source.length() / 2;
	int m = target.length() / 2;
	if (m == 0) return n;
	if (n == 0) return m;
	//Construct a matrix  
	typedef vector< vector<int> >  Tmatrix;
	Tmatrix matrix(n + 1);
	for (int i = 0; i <= n; i++)  matrix[i].resize(m + 1);

	//step 2 Initialize  

	for (int i = 1; i <= n; i++) matrix[i][0] = i;
	for (int i = 1; i <= m; i++) matrix[0][i] = i;

	//step 3  
	char q[40];
	strcpy(q, source.c_str());
	char p[40];
	strcpy(p, target.c_str());
	for (int i = 1; i <= n; i++)
	{
		string temp = "";
		for (int j = 0; j < 2; j++)
		{
			temp += q[i - 1 + j];
		}
		//const char si = source[i - 1];
		//step 4  
		for (int j = 1; j <= m; j++)
		{
			string temp2 = "";
			for (int j = 0; j < 2; j++)
			{
				temp2 += p[i - 1 + j];
			}
			//const char dj = target[j - 1];
			//step 5  
			int cost;
			if (temp == temp2){//copy
				cost = 0;
			}
			else{//submit
				cost = 1;
			}
			//step 6  
			const int above = matrix[i - 1][j] + 1;//delete
			const int left = matrix[i][j - 1] + 1;//insert
			const int diag = matrix[i - 1][j - 1] + cost;
			matrix[i][j] = min(above, min(left, diag));

		}
	}//step7  
	return matrix[n][m];
}
int ldistance2(const vector<string> source, const vector<string> target)
{
	int n = source.size();
	int m = target.size();
	if (m == 0) return n;
	if (n == 0) return m;
	//Construct a matrix  
	string str_source = "", str_target = "";
	typedef vector< vector<int> >  Tmatrix;
	Tmatrix matrix(n + 1);
	for (int i = 0; i <= n; i++)  matrix[i].resize(m + 1);

	//step 2 Initialize  

	for (int i = 1; i <= n; i++) matrix[i][0] = 2 * i;
	for (int i = 1; i <= m; i++) matrix[0][i] = 2 * i;

	//step 3  
	for (int i = 1; i <= n; i++)
	{
		string si = source[i - 1];
		//step 4  
		for (int j = 1; j <= m; j++)
		{

			string  dj = target[j - 1];
			//step 5  
			//int dist=ldistance(si,dj);
			int cost = ldistance(si, dj);

			//step 6  
			const int above = matrix[i - 1][j] + 2;//delete
			const int left = matrix[i][j - 1] + 2;//insert
			const int diag = matrix[i - 1][j - 1] + cost;
			matrix[i][j] = min(above, min(left, diag));
		}
	}//step7  
	//int ma = 0;
	//if (m > n)
	//	ma = m;
	//else
	//	ma = n;

	return matrix[n][m];
}

pair<double, string>  modify_result(string result)
{
	if (result == "")
		return make_pair(0, "");
	int num = sizeof(dictionary) / sizeof(dictionary[0]);
	string min_str = "";
	double min = 10000;

	for (int i = 0; i <num; i++)
	{
		int ma = 0;
		int m = (dictionary[i].length() / 2 - 1) * 2;
		int n = (result.length() / 2 - 1) * 2;
		if (m >n)
			ma = m;
		else
			ma = n;
		int tem = ldistance2(split_words(dictionary[i]), split_words(result));
		double te = double(tem / double(ma));

		if (min>te)
		{
			min = te;
			min_str = dictionary[i];
		}
	}

	return make_pair(min, min_str);
}

pair<double, string>  modify_result2(string result)
{
	if (result == "")
		return make_pair(0, "");
	int num = sizeof(dictionary) / sizeof(dictionary[0]);
	string min_str = "";
	double min = 10000;
	vector<int>min_bin;
	vector<double>min_bin2;
	for (int i = 0; i <num; i++)
	{
		int ma = 0;
		int m = (dictionary[i].length() / 2 - 1) * 2;
		int n = (result.length() / 2 - 1) * 2;
		if (m >n)
			ma = m;
		else
			ma = n;
		int tem = ldistance2(split_words(dictionary[i]), split_words(result));
		double te = double(tem / double(ma));
		min_bin.push_back(tem);
		min_bin2.push_back(te);
		if (min>tem)
		{
			min = tem;
			min_str = dictionary[i];
		}

	}
	sort(min_bin2.begin(), min_bin2.end(), less<double>());

	/*if (min_bin[0] == 1)*/
	if ((min_bin2[0] == 0) || (min_bin[1] != min))
		return make_pair(0, min_str);
	return make_pair(min, min_str);
}

Rect mergeRect(Rect x1, Rect x2)
{
	int left_top_x = x1.x;
	int left_top_y = x1.y;
	int right_bottom_x = x1.x + x1.width;
	int right_bottom_y = x1.y + x1.height;

	if (left_top_x > x2.x)
	{
		left_top_x = x2.x;
	}
	if (left_top_y > x2.y)
	{
		left_top_y = x2.y;
	}
	if (right_bottom_x < x2.x + x2.width)
	{
		right_bottom_x = x2.x + x2.width;
	}
	if (right_bottom_y < x2.y + x2.height)
	{
		right_bottom_y = x2.y + x2.height;
	}

	Rect candidate = Rect(left_top_x, left_top_y, right_bottom_x - left_top_x, right_bottom_y - left_top_y);
	return candidate;
}

void modifyRect(Rect &rect, Mat img){

	if (rect.x < 0)
	{
		rect.width = (rect.width + rect.x);
		rect.x = 0;
	}
	if (rect.y < 0)
	{
		rect.height = (rect.height + rect.y);
		rect.y = 0;
	}
	if (rect.x + rect.width > img.cols)
	{
		rect.width = img.cols - rect.x;
	}
	if (rect.y + rect.height> img.rows)
	{
		rect.height = img.rows - rect.y;
	}
}

Rect expandRectV(Rect_<int> rect, Mat src_cp, double multi)
{

	int expandy = multi*rect.height;
	Mat im = src_cp(rect);
	if (rect.x<0)
	{
		rect.width = (rect.width + rect.x) > src_cp.cols ? src_cp.cols : (rect.width + rect.x);
		rect.x = 0;
	}
	if (rect.y - expandy <= 0)
	{

		rect.height = (rect.height + rect.y + expandy) > src_cp.rows ? src_cp.rows : (rect.height + rect.y + expandy);
		rect.y = 0;
	}
	if (rect.y - expandy > 0)
	{
		rect.y = rect.y - expandy;
		rect.height = (rect.height + 2 * expandy) > src_cp.rows ? src_cp.rows : (rect.height + 2 * expandy);

	}
	if (rect.x + rect.width > src_cp.cols)
	{
		rect.width = src_cp.cols - rect.x;
	}
	if (rect.y + rect.height> src_cp.rows)
	{
		rect.height = src_cp.rows - rect.y;
	}
	//Mat imm = src_cp(rect);
	Rect m = rect;
	return m;

}

pair<double, string> dfs(bool &flag, vector<std::map<double, std::string, std::greater<double> > > result_line, int x, pair<double, string> cur_str, pair<double, string>&min_str)
{
	if (flag)
		return min_str;

	if (x == result_line.size())
	{

		if (min_str.second == "" || cur_str.first < min_str.first)
			min_str = cur_str;

		/*	cur_str.first = 0;
		cur_str.second = "";*/

		if (modify_result2(min_str.second).first == 0)
			flag = true;

		return min_str;
	}

	flag = false;
	pair<double, string>str;
	//for (int i = x; i < result_line.size(); i++)
	int i = x;
	{
		std::map<double, std::string, std::greater<double> >cur_cha = result_line[i];

		std::map<double, std::string, std::greater<double> >::reverse_iterator it = cur_cha.rbegin();
		for (int j = 0; j < 3; j++)
		{
			pair<double, string> tmp_str = cur_str;
			double value = it->first;
			string cha = it->second;

			tmp_str.first += value;
			tmp_str.second += cha;
			str = dfs(flag, result_line, x + 1, tmp_str, min_str);
			it++;
		}
		cur_str;

	}
	return str;

}

string result_end(vector<string>result_terminal, vector<vector<std::map<double, std::string, std::greater<double> > > >result_all)
{

	string result = "";
	pair<double, string>results;
	for (int num_result = 0; num_result < result_terminal.size(); num_result++)
	{
		pair<double, string> s = modify_result2(result_terminal[num_result]);
		if (s.first == 0)
		{
			result = s.second;
			break;
		}
		else
		{
			vector<std::map<double, std::string, std::greater<double> > > result_line = result_all[num_result];

			for (int num_cha = 0; num_cha < result_line.size(); num_cha++)
			{
				std::map<double, std::string, std::greater<double> >::reverse_iterator it = result_line[num_cha].rbegin();

			}
			pair<int, string>end = make_pair(100, "");
			bool flag = false;
			pair<double, string> cur_str;
			pair<double, string>min_str;
			min_str.first = 10000;
			min_str.second = "";
			results = dfs(flag, result_line, 0, cur_str, min_str);
			result = modify_result(results.second).second;

		}
	}
	return result;
}

string rec_img(Mat img, string mqdf_load_path)
{
	double thresh = 125;
	double hw_lowRatio = 0.45;
	double hw_highRatio = 1.45;
	int num_pic = 0;

	resize(img, img, Size(), 2, 2, INTER_LINEAR);
	int width_estimate;
	vector<vector<Rect_<int> > > rect = Bbox(img, width_estimate,mqdf_load_path);
	vector<string>result_terminal;
	vector<vector<std::map<double, std::string, std::greater<double> > > >result_all;
	for (int num = 0; num < rect.size(); num++)
	{
		vector<std::map<double, std::string, std::greater<double> > > result_line;
		vector<Rect_<int> >rect2 = rect[num];

		vector<Rect_<int> >::iterator itt = rect2.begin();
		for (int i = 0; i < rect2.size(); i++)
		{
			if (rect2[i].width == 0)
			{
				rect2.erase(itt + i);
				i--;
			}
		}

		vector<Rect> result_rect;
		string ch = "";
		int m_indx = 0;


		double min_value;
		for (int index1 = 0; index1 < rect2.size(); index1++)
		{
			vector<Rect>::iterator it = rect2.begin();
			Rect x1 = rect2[index1];
			modifyRect(x1, img);

			vector<pair<Rect, int> >Rect_index;
			Rect_index.push_back(make_pair(x1, index1));
			for (int index2 = index1 + 1; index2 < rect2.size(); index2++)
			{
				modifyRect(rect2[index2], img);
				Rect x2 = mergeRect(x1, rect2[index2]);
				Mat imm = img(x2);
				if (x2.width >width_estimate*1.35)
					break;
				Rect_index.push_back(make_pair(x2, index2));
				x1 = x2;
			}
			double min_value = 1000;
			int min_index = 0;
			pair<Rect, int> min_Rect_index = Rect_index[0];
			string min_cha = "";
			std::map<double, std::string, std::greater<double> >single_cha;
			int rank = 3;
			for (int m_rect = 0; m_rect < Rect_index.size(); m_rect++)
			{

				//Mat temp_im2 = img(Rect_index[m_rect].first);
				//std::map<double, std::string, std::greater<double> > Istext2 = NewSample(4, rank, temp_im2, mqdf_load_path);
				//int mi = min(Rect_index[m_rect].first.width, Rect_index[m_rect].first.height);
				//int expand = cvRound(0.05*mi);

				Rect temp_rect = expandRectV(Rect_index[m_rect].first, img, 0.1);
				Mat temp_im1 = img(temp_rect);
				std::map<double, std::string, std::greater<double> > Istext = NewSample(4, rank, temp_im1, mqdf_load_path);
				for (auto it = Istext.begin(); it != Istext.end(); ++it)
				{
					char utf8_res[OUTLEN];
					// cout << it->second << endl;
					g2u((char*)it->second.c_str(), strlen(it->second.c_str()), utf8_res, OUTLEN);
					it->second = string(utf8_res);
					// cout << it->second << endl;
				}


				if (min_value > Istext.rbegin()->first)
				{
					single_cha.clear();
					min_value = Istext.rbegin()->first;
					min_Rect_index = Rect_index[m_rect];
					min_cha = Istext.rbegin()->second;
					single_cha = Istext;
				}

			}

			if (min_value > thresh || (min_Rect_index.first.width < hw_lowRatio*width_estimate))
				//|| (min_Rect_index.first.width > hw_highRatio*width_estimate))
			{
				rect2.erase(it + index1);
				index1--;


				continue;
			}
			else
			{
				result_line.push_back(single_cha);
				Mat temp_cha = img(min_Rect_index.first);
				ch += min_cha;
				index1 = min_Rect_index.second;

			}
		}
		result_terminal.push_back(ch);
		result_all.push_back(result_line);
	}

	string result = result_end(result_terminal, result_all);
	
	return result;
}



PlateRecognizor::PlateRecognizor(std::string mqdf_model_path) : 
					dict({"被告", "被告人", "被上诉人", 
  					   "辩护人", "代理人", "第三人", 
  					   "法定代理人", "上诉人", "公诉人", 
  					   "检察机关", "检察员", "鉴定人", 
  					   "陪审员", "人民陪审员", "审判员", 
  					   "审判长", "书记员", "诉讼代理人", 
  					   "委托代理人", "原告", "原审第三人", 
  					   "代理审判长", "原告代理人"}), model_path(mqdf_model_path) {
}

PlateRecognizor::~PlateRecognizor() {

}

void PlateRecognizor::SetModel(std::string mqdf_model_path) {
  this->model_path = mqdf_model_path;
}
   

std::string PlateRecognizor::recognit(std::string im_path) { 
  // read image
  Mat img = imread(im_path);
  string res = rec_img(img, this->model_path);
  // char utf8_res[OUTLEN];
  // g2u((char*)gbk_res.c_str(), strlen(gbk_res.c_str()), utf8_res, OUTLEN);
  // string utf8_str(utf8_res);
  std::cout << res;
  return res;
}