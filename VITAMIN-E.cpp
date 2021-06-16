// VITAMIN-E.cpp : Defines the entry point for the application.
//

#include <iostream>
#include <chrono>
#include <fstream>
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "Dataset.h"
#include "VITAMIN-E.h"

using namespace std;

class Frame
{
public:
	Frame(int id, cv::Mat& img) : id_(id), img_(img)
	{
		const int resize_scale = 4;
		cv::resize(img_, ds_img_, cv::Size(img_.cols / resize_scale, img.rows / resize_scale));

		cv::Mat ix, iy, ixx, ixy, iyy, iyiyixx, ixiyixy, ixixiyy;
		cv::Sobel(img_, ix, CV_16S, 1, 0, 3);
		cv::Sobel(img_, iy, CV_16S, 0, 1, 3);
		cv::Sobel(cv::abs(ix), ixx, CV_16S, 1, 0, 3);
		cv::Sobel(cv::abs(ix), ixy, CV_16S, 0, 1, 3);
		cv::Sobel(cv::abs(iy), iyy, CV_16S, 0, 1, 3);
		
		iyiyixx = iy.mul(iy.mul(ixx));
		ixiyixy = ix.mul(iy.mul(ixy));
		iyiyixx = ix.mul(ix.mul(iyy));
		kappa_ = cv::abs(iyiyixx - 2 * ixiyixy + iyiyixx);
	}

	void featureExtract(cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor>& brief_ptr)
	{
		const int fast_threshold = 10;
		cv::FAST(ds_img_, ds_kps_, fast_threshold);
		brief_ptr->compute(ds_img_, ds_kps_, ds_desc_);
	}

	int id_;
	cv::Mat img_, ds_img_, ds_desc_, kappa_;
	vector<cv::KeyPoint> ds_kps_;
	Sophus::SE3f T_i_w;
};

int main()
{
	string icl_nuim_lr_kt2 = "C:/Dataset/ICL-NUIM/living_room_traj2_frei_png/";
	// ICLNUIMDataset dataset(icl_nuim_lr_kt2, ICLNUIMDataset::ICLNUIM::LR_KT2);

	string rgbd_freiburg1_xyz = "C:/Dataset/TUMRGB-D/rgbd_dataset_freiburg1_xyz/";
	// TUMRGBDDataset dataset(rgbd_freiburg1_xyz, TUMRGBDDataset::TUMRGBD::FREIBURG1);
	
	string tsukuba = "C:/Dataset/NewTsukubaStereoDataset/";
	// NewStereoTsukubaDataset dataset(tsukuba);

	string kitti = "C:/Dataset/KITTI/data_odometry_gray/";
	KITTIDataset dataset(kitti, 0);

	vector<Frame> frames;
	frames.reserve(dataset.size());
	auto brief_ptr = cv::xfeatures2d::BriefDescriptorExtractor::create(32, true);
	auto matcher = cv::BFMatcher::create(cv::BFMatcher::BRUTEFORCE_HAMMINGLUT, true);
	bool flag_initialize = true;
	double sum = 0.;
	for (int index = 0; index < dataset.size(); ++index)
	{
		chrono::system_clock::time_point start, end;
		start = chrono::system_clock::now();
		
		// Read Image
		cv::Mat img;
		dataset.getData(index, img);
		Frame frame(index, img);

		if (flag_initialize)
		{

		}


		end = chrono::system_clock::now();
		double process_time = static_cast<double>(chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0);
		sum += process_time;
		cout << process_time << "[ms] : " << ((index + 1.)/ sum) * 1000.f << "[fps]\n";
	}
	return 0;
}
