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

#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>
#include <opengv/sac_problems/relative_pose/EigensolverSacProblem.hpp>

typedef Eigen::Matrix<double, 6, 1> Vector6f;
typedef Eigen::Matrix<double, 7, 1> Vector7f;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 7, 1> Vector7d;


static inline double getSubpixelValue(const uchar *img_ptr, const double x, const double y, const int width, const int height)
{
    const int x0 = floor(x), y0 = floor(y);
    const int x1 = x0 + 1, y1 = y0 + 1;
    
    const double x1_weight = x - x0;
    const double y1_weight = y - y0;
    const double x0_weight = 1. - x1_weight;
    const double y0_weight = 1. - y1_weight;

    const double weight_00 = x0_weight * y0_weight;
    const double weight_10 = x1_weight * y0_weight;
    const double weight_01 = x0_weight * y1_weight;
    const double weight_11 = x1_weight * y1_weight;

    const double sum_weight = weight_00 + weight_01 + weight_10 + weight_11;

    double total = (double)img_ptr[y0 * width + x0] * weight_00 
        + (double)img_ptr[y0 * width + x1] * weight_01 
        + (double)img_ptr[y1 * width + x0] * weight_10
        + (double)img_ptr[y1 * width + x1] * weight_11;
    
    return total / sum_weight;
}


struct Camera
{
    typedef std::shared_ptr<Camera> Ptr;
    int width_, height_;
	double fx_, fy_, cx_, cy_;
    cv::Matx<double, 1, 5> distortions_;
	cv::Matx33d K_ = cv::Matx33d::eye();

    Camera() {}
	
	Camera(double fx, double fy, double cx, double cy) : fx_(fx), fy_(fy), cx_(cx), cy_(cy)
	{
		K_(0, 0) = fx;
		K_(1, 1) = fy;
		K_(0, 2) = cx;
		K_(1, 2) = cy;
	}

    Camera(int width, int height, double fx, double fy, double cx, double cy)
        : width_(width), height_(height), fx_(fx), fy_(fy), cx_(cx), cy_(cy)
    {
        K_(0, 0) = fx;
		K_(1, 1) = fy;
		K_(0, 2) = cx;
		K_(1, 2) = cy;
    }

	Camera(cv::Matx33f &K) : K_(K)
	{
		fx_ = K(0, 0);
		fy_ = K(1, 1);
		cx_ = K(0, 2);
		cy_ = K(1, 2);
	}

    inline void setDistortions(double d0, double d1, double d2, double d3, double d4=0)
    {
        distortions_(0) = d0;
        distortions_(1) = d1;
        distortions_(2) = d2;
        distortions_(3) = d3;
        distortions_(4) = d4;
    }

    cv::Mat undistort(cv::Mat &img)
    {
        cv::Mat undistorted;
        cv::undistort(img, undistorted, K_, distortions_);
        return undistorted;
    }

    inline Eigen::Vector2d project(const Eigen::Vector3d &p3d)
    {
        Eigen::Vector2d res;
        res[0] = (fx_ * p3d[0]) / p3d[2] + cx_;
        res[1] = (fy_ * p3d[1]) / p3d[2] + cy_;
        return res;
    }

    inline Eigen::Vector3d unproject(const Eigen::Vector2d &p2d)
    {
        Eigen::Vector3d res;
        res[0] = (p2d[0] - cx_) / fx_;
        res[1] = (p2d[1] - cy_) / fy_;
        res[2] = 1;
        res.normalize();
        return res;
    }
};


struct MeasureTime
{
    inline MeasureTime(std::string name) : name(name)
    {
        start();
    }

    inline void start()
    {
        printf("[TIME] : [START] %s\n", name.c_str());
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
        printf("[Time] : [ END ] %s %s %f[ms]\n", name.c_str(), s.c_str(), duration);
        return duration;
    }

private:
    std::chrono::system_clock::time_point start_, end_;
    std::string name;
};


class Frame;


struct Feature
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Feature> Ptr;

    // std::weak_ptr<Frame> frame_;
    Eigen::Vector2d position_;

    bool is_outlier_ = false;

    Feature() {}
    // Feature(std::shared_ptr<Frame> frame, const Eigen::Vector2d &position) : frame_(frame), position_(position) {}
    Feature(const Eigen::Vector2d &position) : position_(position) {}
};


struct MatchData
{
public:
    std::vector<std::pair<unsigned int, unsigned int>> matches_;
    std::vector<unsigned int> inliers_;
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


class MapPoint
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<MapPoint> Ptr;
    unsigned int id_ = 0;
    Eigen::Vector3d position_ = Eigen::Vector3d::Zero();
    int observed_times_ = 0;
    std::list<std::weak_ptr<Feature>> observations_;

    MapPoint() {}
    MapPoint(int id, Eigen::Vector3d position) : id_(id), position_(position) {}

    void addObservation(Feature::Ptr feature)
    {
        observations_.push_back(feature);
        ++observed_times_;
    }

    static MapPoint::Ptr createNewMapPoint()
    {
        static long factory_id = 0;
        MapPoint::Ptr new_mappoint = std::make_shared<MapPoint>();
        new_mappoint->id_ = factory_id++;
        return new_mappoint;
    }
};


class Map
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Map> Ptr;
    typedef std::unordered_map<unsigned int, MapPoint::Ptr> LandmarksType;
    typedef std::unordered_map<unsigned int, Frame::Ptr> KeyframesType;
    
    Map() {}

    void insertKeyFrame(Frame::Ptr frame);
    void insertMapPoint(MapPoint::Ptr map_point);

    LandmarksType getAllMapPoints()
    {
        return landmarks_;
    }

    KeyframesType getAllKeyframes()
    {
        return keyframes_;
    }

    LandmarksType getActiveMapPoints()
    {
        return active_landmarks_;
    }

    KeyframesType getActiveKeyframes()
    {
        return active_keyframes_;
    }

private:
    LandmarksType landmarks_;
    LandmarksType active_landmarks_;
    KeyframesType keyframes_;
    KeyframesType active_keyframes_;

    Frame::Ptr current_frame_ = nullptr;

    int num_active_keyframes_ = 7;
};