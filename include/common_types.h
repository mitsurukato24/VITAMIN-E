#pragma once

#include <iostream>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>
#include <unordered_map>
#include <string>
#include <omp.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

typedef Eigen::Vector2f Vector2f;
typedef Eigen::Vector3f Vector3f;
typedef Eigen::Matrix<float, 6, 1> Vector6f;
typedef Eigen::Matrix<float, 7, 1> Vector7f;
typedef Eigen::Matrix3f Matrix3f;
typedef Eigen::Matrix4f Matrix4f;

typedef Eigen::Vector2d Vector2d;
typedef Eigen::Vector3d Vector3d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 7, 1> Vector7d;
typedef Eigen::Matrix3d Matrix3d;
typedef Eigen::Matrix4d Matrix4d;

typedef Sophus::SE3f SE3f;
typedef Sophus::SO3f SO3f;
typedef Sophus::SE3d SE3d;
typedef Sophus::SO3d SO3d;


static inline double getSubpixelValue(const uchar* img_ptr, const double x, const double y, const int width, const int height)
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


struct Timer
{
    inline Timer(std::string name) : name_(name)
    {
        start();
    }

    inline double print(std::string s="")
    {
        double duration = end();
        if (s == "")
        {
            printf("[Time] : [ END ] %s %f[ms]\n", name_.c_str(), duration);
        }
        else
        {
            printf("[Time] : [ --- ] --- %s %f[ms]\n", s.c_str(), duration);
        }        
        return duration;
    }

private:
    inline void start()
    {
        printf("[TIME] : [START] %s\n", name_.c_str());
        start_ = std::chrono::system_clock::now();
    }

    inline double end()
    {
        // return [ms]
        end_ = std::chrono::system_clock::now();
        return static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end_ - start_).count() / 1000.0);
    }

    std::chrono::system_clock::time_point start_, end_;
    std::string name_;
};


class Frame;
class Landmark;


struct Feature
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Feature> Ptr;

    std::weak_ptr<Frame> frame_;
    Eigen::Vector2d position_;
    std::weak_ptr<Landmark> landmark_;

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
	
    const int resize_scale_ = 4;
	cv::Mat_<uchar> img_, ds_img_, kappa_;
    cv::Mat ds_desc_;
	std::vector<cv::KeyPoint> ds_kps_;
    std::vector<cv::DMatch> ds_matches_;

    std::vector<Feature::Ptr> features_;
    MatchData match_data_;

    std::mutex pose_mutex_;
    Sophus::SE3d T_i_w_;  // world -> this frame

	Frame(unsigned int id, cv::Mat &img) : id_(id), img_(img)
	{
        // Resize for feature matching
		cv::resize(img_, ds_img_, cv::Size(img_.cols / resize_scale_, img.rows / resize_scale_));

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

        cv::normalize(
            cv::abs(iyiyixx - 2 * ixiyixy + iyiyixx), kappa_,
            0, 255, cv::NORM_MINMAX, CV_8UC1
        );
	}

    static std::shared_ptr<Frame> createFrame(cv::Mat &img)
    {
        static int factory_id = 0;
        Frame::Ptr new_frame = std::make_shared<Frame>(factory_id++, img);
        return new_frame;
    }

    void setKeyframe()
    {
        static int keyframe_factory_id = 0;
        is_keyframe_ = true;
        keyframe_id_ = keyframe_factory_id++;
    }

	void featureExtract(const cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> &brief_ptr)
    {
        const int fast_threshold = 15;
        cv::FAST(ds_img_, ds_kps_, fast_threshold);
        brief_ptr->compute(ds_img_, ds_kps_, ds_desc_);
	}

    Sophus::SE3d getPose()
    {
        std::unique_lock<std::mutex> lock(pose_mutex_);
        return T_i_w_;
    }

    void setPose(const Sophus::SE3d T_i_w)
    {
        std::unique_lock<std::mutex> lock(pose_mutex_);
        T_i_w_ = T_i_w;
    }
};


typedef std::unordered_map<unsigned long, Frame::Ptr> FramesType;


class Landmark
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Landmark> Ptr;
    unsigned int id_ = 0;
    bool is_outlier_ = false;
    Eigen::Vector3d position_ = Eigen::Vector3d::Zero();
    std::mutex data_mutex_;
    int observed_times_ = 0;
    std::list<std::weak_ptr<Feature>> observations_;

    Landmark() {}
    Landmark(int id, Eigen::Vector3d position) : id_(id), position_(position) {}

    void addObservation(Feature::Ptr feature)
    {
        std::unique_lock<std::mutex> lock(data_mutex_);
        observations_.push_back(feature);
        ++observed_times_;
    }

    static Landmark::Ptr createNewLandmark()
    {
        static long factory_id = 0;
        Landmark::Ptr new_landmark = std::make_shared<Landmark>();
        new_landmark->id_ = factory_id++;
        return new_landmark;
    }

    void removeObservation(std::shared_ptr<Feature> feat)
    {
        std::unique_lock<std::mutex> lock(data_mutex_);
		for (auto iter = observations_.begin(); iter != observations_.end(); iter++)
		{
			if (iter->lock() == feat) 
			{
				observations_.erase(iter);
				feat->landmark_.reset();
				observed_times_--;
				break;
			}
		}
    }

    std::list<std::weak_ptr<Feature>> getObservation()
    {
        std::unique_lock<std::mutex> lock(data_mutex_);
        return observations_;
    }

    Eigen::Vector3d getPosition()
    {
        std::unique_lock<std::mutex> lock(data_mutex_);
        return position_;
    }

    void setPosition(const Eigen::Vector3d &position)
    {
        std::unique_lock<std::mutex> lock(data_mutex_);
        position_ = position;
    }
};


class Map
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Map> Ptr;
    typedef std::unordered_map<unsigned int, Landmark::Ptr> LandmarksType;
    typedef std::unordered_map<unsigned int, Frame::Ptr> KeyframesType;
    
    Map() {}

    void insertKeyframe(Frame::Ptr frame)
    {
        Timer timer("Insert keyframe");
        current_frame_ = frame;
        if (keyframes_.find(frame->keyframe_id_) == keyframes_.end())
        {
            printf("insert\n");
            keyframes_.insert(std::make_pair(frame->keyframe_id_, frame));
            active_keyframes_.insert(std::make_pair(frame->keyframe_id_, frame));
        }
        else
        {
            printf("move\n");
            keyframes_[frame->keyframe_id_] = std::move(frame);
            active_keyframes_[frame->keyframe_id_] = std::move(frame);
        }
        timer.print();
    }

    void insertLandmark(Landmark::Ptr landmark)
    {
        if (landmarks_.find(landmark->id_) == landmarks_.end())
        {
			landmarks_.insert(make_pair(landmark->id_, landmark));
			active_landmarks_.insert(make_pair(landmark->id_, landmark));
		}
		else
        {
			landmarks_[landmark->id_] = landmark;
			active_landmarks_[landmark->id_] = landmark;
		}
    }

    LandmarksType getAllLandmarks()
    {
        std::unique_lock<std::mutex> lock(data_mutex_);
        return landmarks_;
    }

    KeyframesType getAllKeyframes()
    {
        std::unique_lock<std::mutex> lock(data_mutex_);
        return keyframes_;
    }

    LandmarksType getActiveLandmarks()
    {
        std::unique_lock<std::mutex> lock(data_mutex_);
        return active_landmarks_;
    }

    KeyframesType getActiveKeyframes()
    {
        std::unique_lock<std::mutex> lock(data_mutex_);
        return active_keyframes_;
    }

    void cleanMap()
    {
        int count_landmark_removed = 0;
		for (auto iter = active_landmarks_.begin(); iter != active_landmarks_.end();)
        {
			if (iter->second->observed_times_ == 0)
            {
				iter = active_landmarks_.erase(iter);
				++count_landmark_removed;
			}
			else
            {
				++iter;
			}
		}
		// LOG(INFO) << "Removed " << cnt_landmark_removed << " active landmarks";
    }

private:
    void removeOldKeyframe()
    {
		if (current_frame_ == nullptr) return;

		// find two closest and farthest keyframe from current frame
		double max_dis = 0, min_dis = 9999;
		double max_kf_id = 0, min_kf_id = 0;
		auto T_w_c = current_frame_->getPose().inverse();  // current -> world
		for (auto& kf : active_keyframes_)
        {
			if (kf.second == current_frame_) continue;
			auto dis = (kf.second->getPose() * T_w_c).log().norm();
			if (dis > max_dis)
            {
				max_dis = dis;
				max_kf_id = kf.first;
			}
			if (dis < min_dis) 
            {
				min_dis = dis;
				min_kf_id = kf.first;
			}
		}

		const double min_dis_th = 0.2;
		Frame::Ptr frame_to_remove = nullptr;
		if (min_dis < min_dis_th)
        {
			//  If there is a close frame, delete the closest one first
			frame_to_remove = keyframes_.at(min_kf_id);
		}
		else
        {
			// Delete the farthest one
			frame_to_remove = keyframes_.at(max_kf_id);
		}

		// LOG(INFO) << "remove keyframe " << frame_to_remove->keyframe_id_;

		// remove keyframe and landmark observation
		active_keyframes_.erase(frame_to_remove->keyframe_id_);
		for (auto feat : frame_to_remove->features_)
        {
			auto landmark = feat->landmark_.lock();
			if (landmark)
            {
				landmark->removeObservation(feat);
			}
		}
        /*
		for (auto feat : frame_to_remove->features_)
        {
			if (feat == nullptr) continue;
			auto mp = feat->map_point_.lock();
			if (mp) {
				mp->RemoveObservation(feat);
			}
		}
        */

		cleanMap();
    }

    std::mutex data_mutex_;
    LandmarksType landmarks_;
    LandmarksType active_landmarks_;
    KeyframesType keyframes_;
    KeyframesType active_keyframes_;

    Frame::Ptr current_frame_ = nullptr;

    int num_active_keyframes_ = 7;
};


// forward declaration
template <typename T>
inline void hash_combine(std::size_t& seed, const T& val);

// default to std::hash
template <typename T>
struct pair_hash : public std::hash<T> {};

// specialize for std::pair
template <typename S, typename T>
struct pair_hash<std::pair<S, T>> {
  inline std::size_t operator()(const std::pair<S, T>& val) const noexcept
  {
    std::size_t seed = 0;
    hash_combine(seed, val.first);
    hash_combine(seed, val.second);
    return seed;
  }
};

template <typename T>
inline void hash_combine(std::size_t& seed, const T& val)
{
  static_assert(sizeof(size_t) == sizeof(uint64_t),
                "hash_combine is meant for 64bit size_t");

  const size_t m = 0xc6a4a7935bd1e995;
  const int r = 47;

  pair_hash<T> hasher;
  std::size_t hash = hasher(val);

  hash *= m;
  hash ^= hash >> r;
  hash *= m;

  seed ^= hash;
  seed *= m;

  // Completely arbitrary number, to prevent 0's from hashing to 0.
  seed += 0xe6546b64;
}
