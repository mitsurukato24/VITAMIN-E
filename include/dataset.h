#pragma once

#include <fstream>
#include "common_types.h"


class Dataset
{
public:
	int size() { return size_; }
	Camera::Ptr getCamera() { return cam_; }
	// virtual void getData(int index, cv::Mat &img, cv::Mat &depth) const {}

protected:
	Dataset() {}

	Camera::Ptr cam_;
	int width_, height_;
	double fx_, fy_, cx_, cy_;
	double distortion[5] = { 0 };
	cv::Mat K_;
	double baseline_;  // for stereo

	int size_;  // the number of images

	std::vector<std::vector<std::string>> imgs_filenames_, depths_filenames_;
	std::vector<std::vector<Vector6d>> gt_poses_;

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
		cam_ = std::make_shared<Camera>(640, 480, 615.f, 615.f, 320, 240);
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
			Vector6d pose_center;
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
			Sophus::SO3d xi = Sophus::SO3d::exp(pose_center.tail(3));
			Eigen::Vector3d offset(baseline_ / 2.f, 0.f, 0.f);
			offset = xi.matrix() * offset;
			gt_poses_.push_back(std::vector<Vector6d>{
				Sophus::SE3d(xi, pose_center.head(3) - offset).log(),
					Sophus::SE3d(xi, pose_center.head(3) + offset).log()
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

	void getData(int index, cv::Mat& img, cv::Mat& depth, bool flag_right=false, bool flag_color=false)
	{
		assert(0 <= index && index < size_);
		img = cv::imread(imgs_filenames_[index][flag_right], flag_color);
		getDepth(depths_filenames_[index][flag_right], depth);
	}

	void getData(int index, cv::Mat& img, bool flag_right=false, bool flag_color=false)
	{
		assert(0 <= index && index < size_);
		img = cv::imread(imgs_filenames_[index][flag_right], flag_color);
	}

	Sophus::SE3d getPose(int index, bool flag_right=false)
	{
		assert(0 <= index && index < size_);
		return Sophus::SE3d::exp(gt_poses_[index][flag_right]);
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


// TODO: change associations.txt style 
/// <summary>
/// https://vision.in.tum.de/data/datasets/rgbd-dataset
/// </summary>
class TUMRGBDDataset : public Dataset
{
public:
	enum class TUMRGBD { FREIBURG1, FREIBURG2, FREIBURG3 };

	TUMRGBDDataset(const std::string dataset_dir, TUMRGBD tumrgbd)
	{
		setCameraParameters(tumrgbd);

		std::ifstream ifs(dataset_dir + "associations.txt");
		int count = 0;
		std::string line;
		while (std::getline(ifs, line))
		{
			std::stringstream ss(line);
			Vector7d tmp;
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
			Eigen::Quaterniond q(tmp(6), tmp(3), tmp(4), tmp(5));
			gt_poses_.push_back(std::vector<Vector6d>{Sophus::SE3d(q, tmp.head(3)).log()});
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

	void getData(int index, cv::Mat& img, bool flag_rgb = false)
	{
		assert(0 <= index && index < size_);
		img = cv::imread(imgs_filenames_[index][0], flag_rgb);
	}

	Sophus::SE3d getPose(int index)
	{
		assert(0 <= index && index < size_);
		return Sophus::SE3d::exp(gt_poses_[index][0]);
	}

private:
	void setCameraParameters(TUMRGBD tumrgbd)
	{
		switch (tumrgbd)
		{
		case TUMRGBD::FREIBURG1:
			cam_ = std::make_shared<Camera>(640, 480, 517.3f, 516.5f, 318.6f, 255.3f);
			cam_->setDistortions(0.2624f, -0.9531f, -0.0054f, 0.0026f, 1.1633f);
			break;
		case TUMRGBD::FREIBURG2:
			cam_ = std::make_shared<Camera>(640, 480, 520.9f, 521.0f, 325.1f, 249.7f);
			cam_->setDistortions(0.2312f, -0.7849f, -0.0033f, 0.0001f, 0.9172f);
			break;
		case TUMRGBD::FREIBURG3:
			cam_ = std::make_shared<Camera>(640, 480, 535.4f, 539.2f, 320.1f, 247.6f);
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
		cam_ = std::make_shared<Camera>(640, 480, 481.2f, -480.f, 319.5f, 239.5f);

		std::ifstream ifs(dataset_dir + getPoseFilename(iclnuim));
		std::string line;
		int count = 0;
		while (std::getline(ifs, line))
		{
			std::stringstream ss(line);
			Vector7d tmp;
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
			Eigen::Quaterniond q(tmp(6), tmp(3), tmp(4), tmp(5));
			gt_poses_.push_back(std::vector<Vector6d>{Sophus::SE3d(q, tmp.head(3)).log()});
			++count;
		}
		ifs.close();
		size_ = count;
	}
	
	void getData(int index, cv::Mat& img, cv::Mat& depth, bool flag_color=false)
	{
		assert(0 <= index && index < size_);
		img = cv::imread(imgs_filenames_[index][0], flag_color);
		depth = cv::imread(depths_filenames_[index][0], cv::IMREAD_UNCHANGED);
		depth.convertTo(depth, CV_32F, 1.f / 5000.f);
	}

	void getData(int index, cv::Mat& img, bool flag_color=false)
	{
		assert(0 <= index && index < size_);
		img = cv::imread(imgs_filenames_[index][0], flag_color);
	}

	Sophus::SE3d getPose(int index)
	{
		assert(0 <= index && index < size_);
		return Sophus::SE3d::exp(gt_poses_[index][0]);
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
		setCameraParameters(sequence_index);
		std::string sequence_str = getZeropadStr(sequence_index, 2);
		std::ifstream ifs(odometry_dataset_dir + sequence_str + "/times.txt");
		if (ifs.fail())
		{
			std::cerr << "Failed to load dataset!!! : " + odometry_dataset_dir + sequence_str + "/times.txt\n";
			return;
		}
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

	Sophus::SE3d getPose(int index, bool flag_right=false)
	{
		assert(0 <= index && index < size_);
		return Sophus::SE3d::exp(gt_poses_[index][flag_right]);
	}

private:
	void setCameraParameters(int sequence_index)
	{
		if (sequence_index >= 0 && sequence_index <= 2)
		{
			cam_ = std::make_shared<Camera>(1241, 376, 718.856f, 718.856f, 607.1928f, 185.2157f);
			baseline_ = 386.1448f;  // times fx;
		}
		else if (sequence_index == 3)
		{
			cam_ = std::make_shared<Camera>(1241, 376, 721.5377f, 721.5377f, 609.5593f, 172.854f);
			baseline_ = 387.5744f;  // times fx;
		}
		else if (sequence_index <= 12)
		{
			cam_ = std::make_shared<Camera>(1241, 376, 707.0912f, 707.0912f, 601.8873f, 183.1104f);
			baseline_ = 389.8145f;
		}
		else
		{
			throw std::exception();
		}
	}
};


class EUROCDataset : public Dataset
{
public:
	enum class EUROC
	{
		V1_01, V1_02, V1_03, V2_01, V2_02, V2_03,
		MH_01, MH_02, MH_03, MH_04, MH_05
	};

	EUROCDataset(std::string &euroc_dataset_dir, EUROC euroc)
	{
		cam_ = std::make_shared<Camera>(458.654, 457.296, 367.215, 248.375);
		cam_->setDistortions(-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05, 0);

		std::string dataset_path = euroc_dataset_dir + getString(euroc) + "/mav0/";
		std::string time_stamps_path = dataset_path + "cam0/data.csv";
		std::ifstream ifs_time_stamps(time_stamps_path);
		std::string line;
		int count = 0;
		while(std::getline(ifs_time_stamps, line))
		{
			if (line.size() < 20 || line[0] == '#') continue;
			time_stamps_.push_back(line.substr(0, 19));

			std::string img_filename = line.substr(20, line.size() - 21);
			imgs_filenames_.push_back(std::vector<std::string>{
				dataset_path + "cam0/data/" + img_filename,
				dataset_path + "cam1/data/" + img_filename
			});
			++count;
		}
		size_ = count;
	}

	void getData(int index, cv::Mat &img, bool flag_right=false, bool flag_color=false)
	{
		img = cv::imread(imgs_filenames_[index][flag_right], flag_color);
	}

private:
	std::string getString(EUROC euroc)
	{
		switch (euroc)
		{
		case EUROC::V1_01:
			return "V1_01_easy";
		case EUROC::V1_02:
			return "V1_02_medium";
		case EUROC::V1_03:
			return "V1_03_difficult";
		case EUROC::V2_01:
			return "V2_01_easy";
		case EUROC::V2_02:
			return "V2_02_medium";
		case EUROC::V2_03:
			return "V2_03_difficult";
		case EUROC::MH_01:
			return "MH_01_easy";
		case EUROC::MH_02:
			return "MH_02_easy";
		case EUROC::MH_03:
			return "MH_03_medium";
		case EUROC::MH_04:
			return "MH_04_difficult";
		case EUROC::MH_05:
			return "MH_05_difficult";
		default:
			throw std::exception();
		}
	}
};
