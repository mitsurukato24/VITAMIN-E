#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>

typedef Eigen::Matrix<float, 6, 1> Vector6f;
typedef Eigen::Matrix<float, 7, 1> Vector7f;


struct Camera
{
	float fx, fy, cx, cy;
	float distortion[5];
	cv::Matx33f K = cv::Matx33f::eye();
	
	Camera(float _fx, float _fy, float _cx, float _cy) : fx(_fx), fy(_fy), cx(_cx), cy(_cy)
	{
		K(0, 0) = fx;
		K(1, 1) = fy;
		K(0, 2) = cx;
		K(1, 2) = cy;
	}

	Camera(cv::Matx33f &_K) : K(_K)
	{
		fx = K(0, 0);
		fy = K(1, 1);
		cx = K(0, 2);
		cy = K(1, 2);
	}
};


class Dataset {
public:
	int size() { return size_; }

protected:
	Dataset() {}

	int width_, height_;
	float fx_, fy_, cx_, cy_;
	float distortion[5] = { 0 };
	cv::Mat K_;
	float baseline_;  // for stereo

	int size_;  // the number of images

	std::vector<std::vector<std::string>> imgs_filenames_, depths_filenames_;
	std::vector<std::vector<Vector6f>> gt_poses_;

	std::vector<std::string> time_stamps_;

	inline std::string getZeropadStr(int num, int len)
	{
		// For example: getNumOfZeropadString(1234, 6) return "001234"
		std::ostringstream oss;
		oss << std::setw(len) << std::setfill('0') << num;
		return oss.str();
	}
};


/// <summary>
/// https://home.cvlab.cs.tsukuba.ac.jp/dataset
/// </summary>
class NewStereoTsukubaDataset : public Dataset
{
public:
	enum class ILLUMINATION {DAYLIGHT, FLASHLIGHT, FLUORESCENT, LAMPS};

	NewStereoTsukubaDataset(const std::string dataset_dir, const ILLUMINATION illumination = ILLUMINATION::FLUORESCENT)
	{
		// Set camera parameters
		width_ = 640;
		height_ = 480;
		fx_ = 615.f;
		fy_ = 615.f;
		cx_ = width_ / 2.f;
		cy_ = height_ / 2.f;
		baseline_ = 10.f;
		size_ = 1800;

		const std::string ill_str = getStrOfIllumination(illumination);
		const std::string rgb_dir = dataset_dir + "illumination/" + ill_str + "/";
		const std::string depth_dir = dataset_dir + "groundtruth/depth_maps/";
		const std::string gt_camera_track_filename = dataset_dir + "groundtruth/camera_track.txt";
		std::ifstream ifs(gt_camera_track_filename);
		std::string line;
		int count = 1;
		while (std::getline(ifs, line))
		{
			// Get pose
			Vector6f pose_center;
			std::stringstream ss(line);
			// X Y Z A B C : x, y, z, theta_x, theta_y, theta_z
			for (int i = 0; i < 6; ++i)
			{
				std::string tmps;
				ss >> tmps;
				// rotate 180 deg
				if (i % 3 != 0)
				{
					pose_center(i) = std::stof(tmps);
				}
				else
				{
					pose_center(i) = - std::stof(tmps);
				}
			}
			Sophus::SO3f xi = Sophus::SO3f::exp(pose_center.tail(3));
			Eigen::Vector3f offset(baseline_ / 2.f, 0.f, 0.f);
			offset = xi.matrix() * offset;
			gt_poses_.push_back(std::vector<Vector6f>{
				Sophus::SE3f(xi, pose_center.head(3) - offset).log(),
					Sophus::SE3f(xi, pose_center.head(3) + offset).log()
			});

			// Get filenames
			std::string count_zeropad = getZeropadStr(count, 5);
			imgs_filenames_.push_back(std::vector<std::string>{
				rgb_dir + "left/tsukuba_" + ill_str + "_L_" + count_zeropad + ".png",
					rgb_dir + "right/tsukuba_" + ill_str + "_R_" + count_zeropad + ".png"
			});
			depths_filenames_.push_back(std::vector<std::string>{
				depth_dir + "left/tsukuba_depth_L_" + count_zeropad + ".xml",
					depth_dir + "right/tsukuba_depth_R_" + count_zeropad + ".xml"
			});
			++count;
		}
		ifs.close();
	}

	void getData(int index, cv::Mat& img, cv::Mat& depth, bool flag_right=false, bool flag_rgb=false)
	{
		assert(0 <= index && index < size_);
		img = cv::imread(imgs_filenames_[index][flag_right], flag_rgb);
		getDepth(depths_filenames_[index][flag_right], depth);
	}

	void getData(int index, cv::Mat& img, bool flag_right = false, bool flag_rgb = false)
	{
		assert(0 <= index && index < size_);
		img = cv::imread(imgs_filenames_[index][flag_right], flag_rgb);
	}

	void getPose(int index, Sophus::SE3f& xi, bool flag_right=false)
	{
		assert(0 <= index && index < size_);
		xi = Sophus::SE3f::exp(gt_poses_[index][flag_right]);
	}

private:
	void getDepth(std::string filename, cv::Mat& depth)
	{
		cv::FileStorage fs(filename, cv::FileStorage::READ);
		if (!fs.isOpened()) throw std::exception();
		fs["depth"] >> depth;
		fs.release();
		depth.convertTo(depth, CV_32F, 0.1f);
	}

	inline std::string getStrOfIllumination(ILLUMINATION illumination)
	{
		switch (illumination)
		{
		case ILLUMINATION::DAYLIGHT:
			return "daylight";
		case ILLUMINATION::FLASHLIGHT:
			return "flashlight";
		case ILLUMINATION::FLUORESCENT:
			return "fluorescent";
		case ILLUMINATION::LAMPS:
			return "lamps";
		default:
			throw;
		}
	}
};


/// <summary>
/// https://vision.in.tum.de/data/datasets/rgbd-dataset
/// </summary>
class TUMRGBDDataset : public Dataset
{
public:
	enum class TUMRGBD { FREIBURG1, FREIBURG2, FREIBURG3 };

	TUMRGBDDataset(const std::string dataset_dir, TUMRGBD tumrgbd)
	{
		width_ = 640;
		height_ = 480;
		setCalibrationParameters(tumrgbd);

		std::ifstream ifs(dataset_dir + "associations.txt");
		int count = 0;
		std::string line;
		while (std::getline(ifs, line))
		{
			std::stringstream ss(line);
			Vector7f tmp;
			for (int i = 0; i < 12; ++i)
			{
				std::string s;
				ss >> s;
				if (i == 1)
				{
					imgs_filenames_.push_back(std::vector<std::string>{ dataset_dir + s });
				}
				else if (i == 3)
				{
					depths_filenames_.push_back(std::vector<std::string>{dataset_dir + s});
				}
				else if (i >= 5)
				{
					tmp(i - 5) = std::stof(s);
				}
			}
			Eigen::Quaternionf q(tmp(6), tmp(3), tmp(4), tmp(5));
			gt_poses_.push_back(std::vector<Vector6f>{Sophus::SE3f(q, tmp.head(3)).log()});
			++count;
		}
		ifs.close();
		size_ = count;
	}

	void getData(int index, cv::Mat& img, cv::Mat& depth, bool flag_rgb = false)
	{
		assert(0 <= index && index < size_);
		img = cv::imread(imgs_filenames_[index][0], flag_rgb);
		depth = cv::imread(depths_filenames_[index][0], cv::IMREAD_UNCHANGED);
		depth.convertTo(depth, CV_32F, 1.f / 5000.f);
	}

	void getPose(int index, Sophus::SE3f& xi)
	{
		assert(0 <= index && index < size_);
		xi = Sophus::SE3f::exp(gt_poses_[index][0]);
	}

private:
	void setCalibrationParameters(TUMRGBD tumrgbd)
	{
		switch (tumrgbd)
		{
		case TUMRGBD::FREIBURG1:
			fx_ = 517.3f;
			fy_ = 516.5f;
			cx_ = 318.6f;
			cy_ = 255.3f;
			distortion[0] = 0.2624f;
			distortion[1] = -0.9531f;
			distortion[2] = -0.0054f;
			distortion[3] = 0.0026f;
			distortion[4] = 1.1633f;
			break;
		case TUMRGBD::FREIBURG2:
			fx_ = 520.9f;
			fy_ = 521.0f;
			cx_ = 325.1f;
			cy_ = 249.7f;
			distortion[0] = 0.2312f;
			distortion[1] = -0.7849f;
			distortion[2] = -0.0033f;
			distortion[3] = 0.0001f;
			distortion[4] = 0.9172f;
			break;
		case TUMRGBD::FREIBURG3:
			fx_ = 535.4f;
			fy_ = 539.2f;
			cx_ = 320.1f;
			cy_ = 247.6f;
			break;
		default:
			throw std::exception();
		}
	}
};


/// <summary>
/// https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html
/// </summary>
class ICLNUIMDataset : public Dataset
{
public:
	enum class ICLNUIM
	{
		LR_KT0,  // living room
		LR_KT1,
		LR_KT2,
		LR_KT3,
		OF_KT0,  // office room
		OF_KT1,
		OF_KT2,
		OF_KT3,
	};

	ICLNUIMDataset(std::string dataset_dir, ICLNUIM iclnuim)
	{
		fx_ = 481.2f;
		fy_ = -480.f;
		cx_ = 319.5f;
		cy_ = 239.5f;

		std::ifstream ifs(dataset_dir + getPoseFilename(iclnuim));
		std::string line;
		int count = 0;
		while (std::getline(ifs, line))
		{
			std::stringstream ss(line);
			Vector7f tmp;
			for (int i = 0; i < 8; ++i)
			{
				std::string s;
				ss >> s;
				if (i == 0)
				{
					time_stamps_.push_back(s);
					imgs_filenames_.push_back(std::vector<std::string>{dataset_dir + "rgb/" + s + ".png"});
					depths_filenames_.push_back(std::vector<std::string>{dataset_dir + "depth/" + s + ".png"});
				}
				else
				{
					tmp(i - 1) = std::stof(s);
				}
			}
			Eigen::Quaternionf q(tmp(6), tmp(3), tmp(4), tmp(5));
			gt_poses_.push_back(std::vector<Vector6f>{Sophus::SE3f(q, tmp.head(3)).log()});
			++count;
		}
		ifs.close();
		size_ = count;
	}
	
	void getData(int index, cv::Mat& img, cv::Mat& depth, bool flag_rgb=false)
	{
		assert(0 <= index && index < size_);
		img = cv::imread(imgs_filenames_[index][0], flag_rgb);
		depth = cv::imread(depths_filenames_[index][0], cv::IMREAD_UNCHANGED);
		depth.convertTo(depth, CV_32F, 1.f / 5000.f);
	}

	void getPose(int index, Sophus::SE3f& xi)
	{
		assert(0 <= index && index < size_);
		xi = Sophus::SE3f::exp(gt_poses_[index][0]);
	}

private:
	std::string getPoseFilename(ICLNUIM iclnuim)
	{
		switch (iclnuim)
		{
		case ICLNUIM::LR_KT0:
			return "livingRoom0.gt.freiburg";
		case ICLNUIM::LR_KT1:
			return "livingRoom1.gt.freiburg";
		case ICLNUIM::LR_KT2:
			return "livingRoom2.gt.freiburg";
		case ICLNUIM::LR_KT3:
			return "livingRoom3.gt.freiburg";
		case ICLNUIM::OF_KT0:
			return "traj0.gt.freiburg";
		case ICLNUIM::OF_KT1:
			return "traj1.gt.freiburg";
		case ICLNUIM::OF_KT2:
			return "traj2.gt.freiburg";
		case ICLNUIM::OF_KT3:
			return "traj3.gt.freiburg";
		default:
			throw std::exception();
		}
	}
};


// TODO: read data_odometry_pose and get gt pose
/// <summary>
/// http://www.cvlibs.net/datasets/kitti/eval_odometry.php
/// </summary>
class KITTIDataset : public Dataset
{
public:
	/// <summary>
	/// KITTI odometry Dataset
	/// </summary>
	/// <param name="dataset_dir">directory path of data_odometry_gray</param>
	/// <param name="sequence_index"> 0 ~ 12</param>
	KITTIDataset(std::string odometry_dataset_dir, int sequence_index = 0)
	{
		assert(sequence_index >= 0 && sequence_index <= 12);
		width_ = 1241;
		height_ = 376;

		std::string sequence_str = getZeropadStr(sequence_index, 2);
		std::ifstream ifs(odometry_dataset_dir + sequence_str + "/times.txt");
		std::string line;
		int count = 0;
		while (std::getline(ifs, line))
		{
			time_stamps_.push_back(line);

			std::string count_zeropad = getZeropadStr(count, 6);
			imgs_filenames_.push_back(std::vector<std::string>{
				odometry_dataset_dir + sequence_str + "/image_0/" + count_zeropad + ".png", 
					odometry_dataset_dir + sequence_str + "/image_1/" + count_zeropad + ".png"
			});
			++count;
		}
		ifs.close();
		size_ = count;
	}

	void getData(int index, cv::Mat &gray, bool flag_right=false)
	{
		gray = cv::imread(imgs_filenames_[index][flag_right], cv::IMREAD_GRAYSCALE);
	}

private:
	void getCameraParameters(int sequence_index)
	{
		if (sequence_index >= 0 && sequence_index <= 2)
		{
			fx_ = 718.856f;
			fy_ = 718.856f;
			cx_ = 607.1928f;
			cy_ = 185.2157f;
			baseline_ = 386.1448f;  // times fx;
		}
		else if (sequence_index == 3)
		{
			fx_ = 721.5377f;
			fy_ = 721.5377f;
			cx_ = 609.5593f;
			cy_ = 172.854f;
			baseline_ = 387.5744f;  // times fx;
		}
		else if (sequence_index <= 12)
		{
			fx_ = 707.0912f;
			fy_ = 707.0912f;
			cx_ = 601.8873f;
			cy_ = 183.1104f;
			baseline_ = 389.8145f;
		}
		else
		{
			throw std::exception();
		}
	}
};