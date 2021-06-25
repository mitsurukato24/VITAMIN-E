#pragma once

#include <iostream>
#include <chrono>
#include <vector>
#include <unordered_map>
#include <string>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

typedef Eigen::Matrix<float, 6, 1> Vector6f;
typedef Eigen::Matrix<float, 7, 1> Vector7f;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 7, 1> Vector7d;


static inline float getSubpixelValue(const uchar *img_ptr, const float x, const float y, const int width, const int height)
{
    const int x0 = floor(x), y0 = floor(y);
    const int x1 = x0 + 1, y1 = y0 + 1;
    
    const float x1_weight = x - x0;
    const float y1_weight = y - y0;
    const float x0_weight = 1.f - x1_weight;
    const float y0_weight = 1.f - y1_weight;

    const float weight_00 = x0_weight * y0_weight;
    const float weight_10 = x1_weight * y0_weight;
    const float weight_01 = x0_weight * y1_weight;
    const float weight_11 = x1_weight * y1_weight;

    const float sum_weight = weight_00 + weight_01 + weight_10 + weight_11;

    float total = (float)img_ptr[y0 * width + x0] * weight_00 
        + (float)img_ptr[y0 * width + x1] * weight_01 
        + (float)img_ptr[y1 * width + x0] * weight_10
        + (float)img_ptr[y1 * width + x1] * weight_11;
    
    return total / sum_weight;
}


struct Camera
{
    int width, height;
	float fx, fy, cx, cy;
	float distortions[5] = {0};
	cv::Matx33f K = cv::Matx33f::eye();

    Camera() {}
	
	Camera(float fx, float fy, float cx, float cy) : fx(fx), fy(fy), cx(cx), cy(cy)
	{
		K(0, 0) = fx;
		K(1, 1) = fy;
		K(0, 2) = cx;
		K(1, 2) = cy;
	}

    Camera(int width, int height, float fx, float fy, float cx, float cy)
        : width(width), height(height), fx(fx), fy(fy), cx(cx), cy(cy)
    {
        K(0, 0) = fx;
		K(1, 1) = fy;
		K(0, 2) = cx;
		K(1, 2) = cy;
    }

	Camera(cv::Matx33f &K) : K(K)
	{
		fx = K(0, 0);
		fy = K(1, 1);
		cx = K(0, 2);
		cy = K(1, 2);
	}

    void setDistortions(float d0, float d1, float d2, float d3, float d4=0)
    {
        distortions[0] = d0;
        distortions[1] = d1;
        distortions[2] = d2;
        distortions[3] = d3;
        distortions[4] = d4;
    }
};


struct MeasureTime
{
    inline MeasureTime()
    {
        start();
    }

    inline void start()
    {
        start_ = std::chrono::system_clock::now();
    }

    inline double end()
    {
        // return [ms]
        end_ = std::chrono::system_clock::now();
        return static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end_ - start_).count() / 1000.0);
    }

    inline double printTime(std::string s="")
    {
        double duration = end();
        printf("[Time] %s: %f[ms]\n", s.c_str(), duration);
        return duration;
    }

private:
    std::chrono::system_clock::time_point start_, end_;
};


class Frame;


struct Feature
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Feature> Ptr;

    // std::weak_ptr<Frame> frame_;
    Eigen::Vector2f position_;

    bool is_outlier_ = false;

    Feature() {}
    // Feature(std::shared_ptr<Frame> frame, const Eigen::Vector2f &position) : frame_(frame), position_(position) {}
    Feature(const Eigen::Vector2f &position) : position_(position) {}
};


struct MatchData
{
public:
    std::vector<std::pair<unsigned int, unsigned int>> matches_;
};


class Frame
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Frame> Ptr;
    unsigned int id_ = 0;
    unsigned int keyframe_id_ = 0;
    bool is_keyframe_ = false;
	Sophus::SE3f T_i_w;  // world -> this frame

	cv::Mat img_, ds_img_, ds_desc_, kappa_;
	std::vector<cv::KeyPoint> ds_kps_;
    std::vector<cv::DMatch> ds_matches_;

    std::vector<Feature::Ptr> features_;
    MatchData match_data_;

	Frame(unsigned int id, cv::Mat& img) : id_(id), img_(img)
	{
        // Resize for feature matching
		const int resize_scale = 4;
		cv::resize(img_, ds_img_, cv::Size(img_.cols / resize_scale, img.rows / resize_scale));

        // Compute curvature image
		cv::Mat blured, ix, iy, ixx, ixy, iyy, iyiyixx, ixiyixy, ixixiyy;
        cv::blur(img_, blured, cv::Size(5, 5));
        cv::Sobel(blured, ix, CV_16S, 1, 0, 3);
        cv::Sobel(blured, iy, CV_16S, 0, 1, 3);
        cv::Sobel(ix, ixx, CV_16S, 1, 0, 3);
        cv::Sobel(ix, ixy, CV_16S, 0, 1, 3);
        cv::Sobel(iy, iyy, CV_16S, 0, 1, 3);
        
		iyiyixx = iy.mul(iy.mul(ixx));
		ixiyixy = ix.mul(iy.mul(ixy));
		iyiyixx = ix.mul(ix.mul(iyy));
		kappa_ = cv::abs(iyiyixx - 2 * ixiyixy + iyiyixx);
        cv::normalize(kappa_, kappa_, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	}

    static std::shared_ptr<Frame> CreateFrame(cv::Mat &img)
    {
        static int factory_id = 0;
        Frame::Ptr new_frame = std::make_shared<Frame>(factory_id++, img);
        // Frame::Ptr new_frame(new Frame(factory_id++, img));
        return new_frame;
    }

    void setKeyFrame()
    {
        static int keyframe_factory_id = 0;
        is_keyframe_ = true;
        keyframe_id_ = keyframe_factory_id++;
    }

	void featureExtract(cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> &brief_ptr)
    {
		const int fast_threshold = 5;
		cv::FAST(ds_img_, ds_kps_, fast_threshold);
		brief_ptr->compute(ds_img_, ds_kps_, ds_desc_);
	}
};