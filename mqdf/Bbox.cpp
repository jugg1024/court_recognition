#include <stdio.h>
#include <list>
#include <vector>
#include "chinese_traverse.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

static int min_text_height = 13;
static int min_vdege_density = 0.08;
static int min_hdege_density = 0.04;
static int min_area = 100;

static float computeDensity(Mat& candidate) {
	Mat binary_img = candidate.clone();
	Sobel(binary_img, binary_img, binary_img.depth(), 1, 1);
	float num_density = 0;
	for (int i = 0; i < binary_img.rows; ++i) {
		for (int j = 0; j < binary_img.cols; ++j) {
			uchar pixel = binary_img.at<uchar>(i, j);
			num_density += pixel;
		}
	}
	int area = binary_img.rows * binary_img.cols;
	num_density /= area;
	return num_density;
}

static int decomposeRegionbyHorizontal(Mat& img, Mat& v_edge, vector< Rect_<int> >& vec_rect) {
	Mat mask_v_edge(v_edge.size(), CV_8UC1, Scalar(0));
	cv::threshold(v_edge, mask_v_edge, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	for (int h = 0; h < v_edge.rows; ++h) {
		uchar* ptr_mask = mask_v_edge.ptr<uchar>(h);
		for (int w = 0; w < v_edge.cols; ++w) {
			if (ptr_mask[w] > 0) {
				ptr_mask[w] = 1;
			}
		}
	}
	Mat integral_img;
	cv::integral(mask_v_edge, integral_img, CV_32SC1);
	vector<int> horizontal_projection_origin(img.rows, 0);
	for (int j = 0; j < img.rows; ++j) {
		horizontal_projection_origin[j] = 0;
		for (int i = 0; i < img.cols; ++i) {
			if (mask_v_edge.at<uchar>(j, i) > 0) {
				horizontal_projection_origin[j] += 1;
			}
		}
	}
	//lj debug
	int space_between_rows = 2;
	//int space_between_rows = 1;
	//lj end
	for (int j = 0; j < mask_v_edge.rows; ++j) {
		int beg_pos = -1;
		int end_pos = -1;
		for (int k = 0; k < mask_v_edge.cols; ++k) {
			unsigned char* ptr_data = mask_v_edge.ptr<unsigned char>(j);
			unsigned char data = ptr_data[k];
			if (data != 0) {
				if (beg_pos == -1) {
					beg_pos = k;
				}
				end_pos = k;
			}
			if (data == 0 || k == mask_v_edge.cols - 1) {
				if (beg_pos != -1) {
					int up_j = j - space_between_rows;
					up_j = up_j < 0 ? 0 : up_j;
					int tlp_x = beg_pos;
					int tlp_y = up_j;
					int brp_x = end_pos + 1;
					int brp_y = j + 1;
					int area = (brp_x - tlp_x) * (brp_y - tlp_y);
					float v_edge_num = integral_img.at<int>(brp_y, brp_x) - integral_img.at<int>(brp_y, tlp_x) - integral_img.at<int>(tlp_y, brp_x) + integral_img.at<int>(tlp_y, tlp_x);
					float dense = v_edge_num / area;
					///////////////////////////
					int bo_j = j + space_between_rows + 1;
					//bo_j = bo_j > mask_v_edge.rows ? mask_v_edge.rows : bo_j;
					tlp_x = beg_pos;
					tlp_y = j + 1;
					brp_x = end_pos + 1;
					brp_y = bo_j + 1;
					brp_y = brp_y > mask_v_edge.rows ? mask_v_edge.rows : brp_y;
					//area = (brp_x - tlp_x + 1) * (brp_y - tlp_y + 1);
					area = (brp_x - tlp_x) * (brp_y - tlp_y);
					float bottom_v_edge_num = integral_img.at<int>(brp_y, brp_x) - integral_img.at<int>(brp_y, tlp_x) - integral_img.at<int>(tlp_y, brp_x) + integral_img.at<int>(tlp_y, tlp_x);
					float bottom_dense = bottom_v_edge_num / area;
					float max_dense = bottom_dense > dense ? bottom_dense : dense;
					if (max_dense < min_vdege_density) {
						for (int i = beg_pos; i < end_pos; ++i) {
							ptr_data[i] = 0;
						}
					}
					beg_pos = -1;
				}
			}
		}
	}
	vector<int> horizontal_projection(img.rows, 0);
	for (int j = 0; j < img.rows; ++j) {
		horizontal_projection[j] = 0;
		for (int i = 0; i < img.cols; ++i) {
			if (mask_v_edge.at<uchar>(j, i) >0) {
				horizontal_projection[j] += 1;
			}
		}
	}
	float thresh = 0;
	int num = 0;
	for (int j = 0; j < img.rows; ++j) {
		int oht = horizontal_projection_origin[j];
		int nht = horizontal_projection[j];
		unsigned char* ptr_data = mask_v_edge.ptr<unsigned char>(j);
		if (oht - nht > oht / 3 && nht != 0) {
			for (int i = 0; i < img.cols; ++i) {
				ptr_data[i] = 0;
			}
			horizontal_projection[j] = 0;
		}
		if (horizontal_projection[j] > 6) {
			thresh += horizontal_projection[j];
			++num;
		}
	}

	//lj debug
	vector<int> horizontal_projection_new(img.rows, 0);
	int span = 2;
	//float ratio = 1.0 / (span * 2);
	float ratio = 1.0 / (span);
	for (int i = span; i < img.rows - span; ++i) {
		horizontal_projection_new[i] = horizontal_projection[i];
		for (int k = 1; k < span; ++k) {
			horizontal_projection_new[i] += ratio*horizontal_projection[i - k];
			horizontal_projection_new[i] += ratio*horizontal_projection[i + k];
		}
	}
	if (num != 0) {
		thresh /= num;
	}
	//lj end
	//thresh /= 2;
	thresh *= 0.88;
	thresh = thresh < 6 ? 6 : thresh;
	int beg_pos = -1;
	int end_pos = -1;
	for (int j = 0; j < img.rows; ++j) {
		//if (horizontal_projection[j] > 0) {
		if (horizontal_projection_new[j] > thresh) {
			if (beg_pos == -1) {
				beg_pos = j;
			}
			end_pos = j;
		}
		//if (horizontal_projection[j] == 0 || j == img.rows - 1) {
		//if (horizontal_projection[j] <= thresh || j == img.rows - 1) {
		if (horizontal_projection_new[j] <= thresh || j == img.rows - 1) {
			if (beg_pos != -1) {
				int height = end_pos - beg_pos + 1;
				if (height < 5) {
					beg_pos = -1;
					continue;
				}
				Rect_<int> rect;
				rect.x = 0;
				rect.y = beg_pos;
				rect.width = img.cols;
				rect.height = height;
				////lj debug
				//int height_extension = height * 0.15 + 0.5;
				//rect.height = height + height_extension + height_extension;
				//rect.y -= height_extension;
				//rect.y = rect.y < 0 ? 0 : rect.y;
				//rect.height = rect.y + rect.height > img.rows ? img.rows-rect.y : rect.height;
				////lj end
				vec_rect.push_back(rect);
				beg_pos = -1;
			}
		}
	}
	return vec_rect.size();
}

static int decomposeRegionbyVertical(Mat& img, Mat& h_edge, Mat& gray, vector< Rect_<int> >& vec_rect, string mqdf_load_path) {
	Mat mask_h_edge(h_edge.size(), CV_8UC1, Scalar(0));
	cv::threshold(h_edge, mask_h_edge, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	for (int h = 0; h < h_edge.rows; ++h) {
		uchar* ptr_mask = mask_h_edge.ptr<uchar>(h);
		uchar* ptr_h_edge = h_edge.ptr<uchar>(h);
		for (int w = 0; w < h_edge.cols; ++w) {
			if (ptr_h_edge[w] > 0) {
				ptr_mask[w] = 1;
			}
		}
	}
	Mat integral_img;
	cv::integral(mask_h_edge, integral_img, CV_32SC1);
	vector<int> vertical_projection_origin(img.cols, 0);
	for (int i = 0; i < img.cols; ++i) {
		vertical_projection_origin[i] = 0;
		for (int j = 0; j < img.rows; ++j) {
			if (mask_h_edge.at<uchar>(j, i) > 0) {
				vertical_projection_origin[i] += 1;
			}
		}
	}
	int space_between_cols = 7;
	for (int k = 0; k < mask_h_edge.cols; ++k) {
		int beg_pos = -1;
		int end_pos = -1;
		for (int j = 0; j < mask_h_edge.rows; ++j) {
			uchar data = mask_h_edge.at<uchar>(j, k);
			if (data != 0) {
				if (beg_pos == -1) {
					beg_pos = j;
				}
				end_pos = j;
			}
			if (data == 0 || j == mask_h_edge.rows - 1) {
				if (beg_pos != -1) {
					int left_k = k - space_between_cols;
					left_k = left_k < 0 ? 0 : left_k;
					int tlp_x = left_k;
					int tlp_y = beg_pos;
					int brp_x = k + 1;
					int brp_y = end_pos + 1;
					int area = (brp_x - tlp_x) * (brp_y - tlp_y);
					float v_edge_num = integral_img.at<int>(brp_y, brp_x) - integral_img.at<int>(brp_y, tlp_x) - integral_img.at<int>(tlp_y, brp_x) + integral_img.at<int>(tlp_y, tlp_x);
					float dense = v_edge_num / area;
					///////////////////////////
					int right_k = k + space_between_cols + 1;
					tlp_x = k + 1;
					tlp_y = beg_pos;
					brp_x = right_k;
					brp_y = end_pos + 1;
					brp_x = brp_x > mask_h_edge.cols ? mask_h_edge.cols : brp_x;
					area = (brp_x - tlp_x) * (brp_y - tlp_y);
					float right_h_edge_num = integral_img.at<int>(brp_y, brp_x) - integral_img.at<int>(brp_y, tlp_x) - integral_img.at<int>(tlp_y, brp_x) + integral_img.at<int>(tlp_y, tlp_x);
					float right_dense = right_h_edge_num / area;
					float max_dense = right_dense > dense ? right_dense : dense;
					if (max_dense < min_hdege_density) {
						for (int i = beg_pos; i < end_pos; ++i) {
							mask_h_edge.at<uchar>(i, k) = 0;
						}
					}
					beg_pos = -1;
				}
			}
		}
	}
	vector<int> vertical_projection(img.cols, 0);
	vector<int> vertical_projection_gray(img.cols, 0);
	for (int i = 0; i < img.cols; ++i) {
		vertical_projection[i] = 0;
		vertical_projection_gray[i] = 0;
		for (int j = 0; j < img.rows; ++j) {
			if (mask_h_edge.at<uchar>(j, i) >0) {
				vertical_projection[i] += 1;
			}
			vertical_projection_gray[i] += h_edge.at<uchar>(j, i);
		}
	}
	int beg_pos = -1;
	int end_pos = -1;
	for (int j = 0; j < img.cols; ++j) {
		if (vertical_projection[j] > 0) {
			if (beg_pos == -1) {
				beg_pos = j;
			}
			end_pos = j;
		}
		if (vertical_projection[j] == 0 || j == img.cols - 1) {
			if (beg_pos != -1) {
				int width = end_pos - beg_pos + 1;
				Rect_<int> rect;
				rect.x = beg_pos;
				rect.y = 0;
				rect.width = width;
				rect.height = img.rows;
				vec_rect.push_back(rect);
				beg_pos = -1;
			}
		}
	}
	if (vec_rect.size() == 0) {
		return 0;
	}
	int height_estimation = vec_rect[0].height;
	float width_estimation = 0;
	int num = 0;
	for (int i = 0; i < vec_rect.size(); ++i) {
		if (vec_rect[i].width > height_estimation*0.7 && vec_rect[i].width < height_estimation*1.2) {
			width_estimation += vec_rect[i].width;
			++num;
		}
	}
	int candidates_num = 0;
	if (num > 1) {
		width_estimation /= num;
	}
	else {
		//lj debug
		width_estimation = 0;
		vector < Rect_<int> > rect_candidates;
		bool first = true;
		int start = 0;
		int stop = 0;
		int step = 2;
		float diff = 2;
		for (int j = 0; j < img.cols; ++j) {
			float min = 9999999;
			float pre_value = vertical_projection_gray[j];
			pre_value = pre_value < 1 ? 1 : pre_value;
			float next_value;
			if (j + step < img.cols) {
				next_value = vertical_projection_gray[j + step];
			}
			else {
				next_value = vertical_projection_gray[img.cols - 1];
			}

			if (next_value / pre_value > diff && start == 0 && stop == 0) {
				start = j;
			}
			if (pre_value / next_value > diff && stop == 0 && (start != 0 || first)) {
				if (j + step < img.cols) {
					stop = j + step;
				}
				else {
					stop = img.cols - 1;
				}
				if (first) {
					Rect_<int> rect(0, 0, stop, img.rows);
					rect_candidates.push_back(rect);
					stop = 0;
					first = false;
				}
				j = stop;
			}
			if (start != 0 && stop != 0 && stop > start) {
				Rect_<int> pre_rect(start, 0, stop - start + 1, img.rows);
				rect_candidates.push_back(pre_rect);
				start = 0;
				stop = 0;
			}
		}
		if (start != 0 && stop != 0 && stop > start) {
			Rect_<int> pre_rect(start, 0, stop - start + 1, img.rows);
			rect_candidates.push_back(pre_rect);
		}
		//lj end
		for (int i = 0; i < rect_candidates.size(); ++i) {
			if (rect_candidates[i].width > height_estimation*0.5 && rect_candidates[i].width < height_estimation*1.2) {
				width_estimation += rect_candidates[i].width;
				++candidates_num;
			}
		}
		if (candidates_num > 0) {
			width_estimation /= candidates_num;
		}
		else {
			width_estimation = height_estimation*0.9;
		}
	}
	list< Rect_<int> > temp_rects;
	for (int i = 0; i < vec_rect.size(); ++i) {
		temp_rects.push_back(vec_rect[i]);
	}
	int start = 0;
	int next_start = 0;
	int stop = 0;
	int step = 2;
	float diff = 2;
	int index = 0;
	bool first = true;
	for (list< Rect_<int> >::iterator iter = temp_rects.begin(); iter != temp_rects.end(); ++iter) {
		if (iter->width > width_estimation*1.2) {
			float min = 9999999;
			Rect_<int> current_rect = *iter;
			for (int x = iter->x + width_estimation*0.7; x < iter->x + width_estimation*1.2; ++x) {
				if (vertical_projection_gray[x] / min < 0.9) {
					//min = vertical_projection[x];
					min = vertical_projection_gray[x];
					if (min < 0.1) {
						min = 0.1;
					}
					index = x;
				}
			}
			Rect_<int> pre_rect(current_rect.x, current_rect.y, index - current_rect.x, current_rect.height);
			Rect_<int> next_rect(index + 1, current_rect.y, current_rect.x + current_rect.width - index - 2, current_rect.height);
			temp_rects.insert(iter, pre_rect);
			*iter = next_rect;
			--iter;
		}
	}
	vec_rect.clear();
	float avearge_density = 0;
	int num_density = 0;
	for (list< Rect_<int> >::iterator iter = temp_rects.begin(); iter != temp_rects.end(); ++iter) {
		if (iter->width > 0) {
			Mat candidate(gray(*iter));
			float density = computeDensity(candidate);
			if (density > 0) {
				avearge_density += density;
				++num_density;
			}
		}
	}
	avearge_density /= num_density;
	int start_1 = 0;
	for (list< Rect_<int> >::iterator iter = temp_rects.begin(); iter != temp_rects.end(); ++iter) {
		if (first && iter->width > 0) {
			float min = 2;
			for (int x = iter->x; x < iter->x + width_estimation; ++x) {
				float pre_value = vertical_projection_gray[x];
				pre_value = pre_value < 1 ? 1 : pre_value;
				float next_value;
				if (x + step < img.cols) {
					next_value = vertical_projection_gray[x + step];
				}
				else {
					next_value = vertical_projection_gray[img.cols - 1];
				}
				//if ((float)next_value / pre_value > min) {
				if ((float)next_value / pre_value > min || (float)next_value / pre_value > 2.5) {
					min = next_value / pre_value;
					min *= 0.9;
					start_1 = x;
				}
			}
		}
		if (start_1 != 0 && first) {
			start_1 -= width_estimation*0.1;
			start_1 = start_1 < 0 ? 1 : start_1;
			for (; iter != temp_rects.end(); ++iter) {
				Rect_<int> current_rect = *iter;
				Rect_<int> next_rect(start_1 + 1, current_rect.y, current_rect.x + current_rect.width - start_1 - 2, current_rect.height);
				if (iter->x > start_1) {
					//*iter = next_rect;
					first = false;
					break;
				}
				else if (iter->x + iter->width > start_1) {
					*iter = next_rect;
					first = false;
					break;
				}
				else {
					iter->width = 0;
				}
			}
			break;
		}
		if (start_1 == 0) {
			iter->width = 0;
		}
	}
	for (list< Rect_<int> >::iterator iter = temp_rects.begin(); iter != temp_rects.end(); ++iter) {
		if (iter->width > 0) {
			Mat candidate(gray(*iter));
			if (computeDensity(candidate) < 0.2)
				continue;
			std::map<double, std::string, std::greater<double> > Istext2 = NewSample(4, 3, candidate, mqdf_load_path);
			if ((computeDensity(candidate) > avearge_density*0.6) || (Istext2.rbegin()->first<130)){
				vec_rect.push_back(*iter);
			}
		}
	}
	return width_estimation;
}

static int decomposeRegion(Mat& gray, Mat& coarse_result, vector< vector<Rect_<int> > >& text_candidates, string mqdf_load_path) {
	int width_estimation = 0;
	Mat mask;
	cv::adaptiveThreshold(gray, mask, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 35, 10);
	int erode_size = 3;
	Mat element = getStructuringElement(MORPH_RECT, Size(2 * erode_size, 2 * erode_size));
	cv::erode(mask, mask, element);
	Mat_<int> labels;
	Mat_<int> stats;
	Mat_<double> centroids;
	int num_comps = connectedComponentsWithStats(coarse_result, labels, stats, centroids);
	for (int i = 1; i < num_comps; ++i) {
		Rect component(stats(i, CC_STAT_LEFT), stats(i, CC_STAT_TOP), stats(i, CC_STAT_WIDTH), stats(i, CC_STAT_HEIGHT));
		//lj debug
		if (component.height < min_text_height || component.area() < min_area) {
			continue;
		}
		vector< Rect_<int> > candidates;
		candidates.push_back(component);
		Mat candidate_region = coarse_result(component);
		Mat v_edge;
		Sobel(gray(component), v_edge, v_edge.depth(), 1, 0);
		//lj debug
		Mat v_edge_1 = v_edge.clone();
		for (int j = 0; j < v_edge.rows; ++j) {
			uchar* mask_data = mask.ptr<uchar>(j);
			uchar* v_edge_data = v_edge_1.ptr<uchar>(j);
			for (int i = 0; i < v_edge.cols; ++i) {
				if (mask_data[i] == 255) {
					v_edge_data[i] = 0;
				}
			}
		}
		v_edge = v_edge_1;
		//lj end
		vector< Rect_<int> > vec_rect;
		decomposeRegionbyHorizontal(candidate_region, v_edge, vec_rect);
		if (vec_rect.size() == 0) {
			return 0;
		}
		int max_height = 0;
		for (int i = 0; i < vec_rect.size(); ++i) {
			Rect_<int> rect = vec_rect[i];
			if (max_height<rect.height) {
				max_height = rect.height;
			}
		}
		for (int i = 0; i < vec_rect.size(); ++i) {
			if (vec_rect[i].height < max_height * 0.7) {
				continue;
			}
			Rect_<int> rect;
			rect.x = component.x + vec_rect[i].x;
			rect.y = component.y + vec_rect[i].y;
			rect.width = vec_rect[i].width;
			rect.height = vec_rect[i].height;
			if (rect.height < min_text_height - 2) {
				continue;
			}
			Mat sub_candidate_region = candidate_region(vec_rect[i]);
			Mat h_edge;
			Sobel(gray(rect), h_edge, h_edge.depth(), 0, 1);
			//lj debug
			Mat mask_h_edge = mask(rect);
			Mat h_edge_1 = h_edge.clone();
			for (int j = 0; j < h_edge.rows; ++j) {
				uchar* mask_data = mask_h_edge.ptr<uchar>(j);
				uchar* h_edge_data = h_edge_1.ptr<uchar>(j);
				for (int i = 0; i < h_edge.cols; ++i) {
					//if (mask_h_edge.at<uchar>(j, i) == 255) {
					if (mask_data[i] == 255) {
						h_edge_data[i] = 0;
					}
				}
			}
			h_edge = h_edge_1;
			//lj end
			vector< Rect_<int> > temp_vec_rect;
			Mat gray_rect = gray(rect);
			width_estimation = decomposeRegionbyVertical(sub_candidate_region, h_edge, gray_rect, temp_vec_rect, mqdf_load_path);
			vector<Rect> text_candidate;
			for (int i = 0; i < temp_vec_rect.size(); ++i) {
				Rect_<int> sub_rect;
				sub_rect.x = rect.x + temp_vec_rect[i].x;
				sub_rect.y = rect.y + temp_vec_rect[i].y;
				sub_rect.width = temp_vec_rect[i].width;
				sub_rect.height = temp_vec_rect[i].height;
				//if (sub_rect.height < min_text_height - 2 || sub_rect.area() < 60) {
				if (sub_rect.height < min_text_height - 2) {
					continue;
				}
				text_candidate.push_back(sub_rect);
			}
			text_candidates.push_back(text_candidate);
		}
	}
	return width_estimation;
}


static int detectText(cv::Mat& img, vector<vector<Rect_<int> > >& text_candidates, string mqdf_load_path) {
	Mat gray;
	if (img.type() == CV_8UC3) {
		cv::cvtColor(img, gray, cv::COLOR_RGB2GRAY);
	}
	else if (img.type() == CV_8UC1) {
		gray = img.clone();
	}
	else {
		return -1;
	}
	//vector< Rect_<int> > text_candidates;
	Mat coarse_result_(Size(gray.cols, gray.rows), CV_8UC1, Scalar(255));
	int width = decomposeRegion(gray, coarse_result_, text_candidates, mqdf_load_path);
	if (text_candidates.size() == 0) {
		return 0;
	}
	/*int left_top_x = text_candidates[0].x;
	int left_top_y = text_candidates[0].y;
	int right_bottom_x = text_candidates[0].x + text_candidates[0].width;
	int right_bottom_y = text_candidates[0].y + text_candidates[0].height;
	for (int i = 1; i < text_candidates.size(); ++i) {
	if (left_top_x > text_candidates[i].x) {
	left_top_x = text_candidates[i].x;
	}
	if (left_top_y > text_candidates[i].y) {
	left_top_y = text_candidates[i].y;
	}
	if (right_bottom_x < text_candidates[i].x + text_candidates[i].width) {
	right_bottom_x = text_candidates[i].x + text_candidates[i].width;
	}
	if (right_bottom_y < text_candidates[i].y + text_candidates[i].height) {
	right_bottom_y = text_candidates[i].y + text_candidates[i].height;
	}
	}
	candidate = Rect(left_top_x, left_top_y, right_bottom_x - left_top_x, right_bottom_y - left_top_y);*/
	return width;
}

vector<vector<Rect_<int> > >  Bbox(Mat im, int &width, string mqdf_load_path) {
	/*int textlen = 0;
	Mat img = imread(".\\pic\\54.jpg");*/
	vector<vector<Rect_<int> > > candidate;
	width = detectText(im, candidate, mqdf_load_path);

	return candidate;
	//	printf("hello world\n");
}


