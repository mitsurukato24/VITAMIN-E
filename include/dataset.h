#pragma once

#include <fstream>
#include "common_types.h"


// depth, baseline, pose are in meter.
class Dataset
{
public:
	int size() { return size_; }
	Camera::Ptr getCamera() { return cam_; }

protected:
	Dataset() {}

	Camera::Ptr cam_;
	int width_, height_;
	float fx_, fy_, cx_, cy_;
	float distortion[5] = { 0 };
	float baseline_;  // for stereo

	int size_;  // the number of images

	std::vector<std::vector<std::string>> imgs_filenames_, depths_filenames_;
	std::vector<std::vector<Vector6f>> gt_poses_;
	std::vector<std::string> time_stamps_;

	inline std::string getZeropadStr(const int num, const int len)
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
		cam_ = std::make_shared<Camera>(640, 480, 615.f, 615.f, 320.f, 240.f);
		baseline_ = 10.f * rescale_;
		size_ = 1800;

		const std::string ill_str = getStrOfIllumination(illumination);
		const std::string rgb_dir = dataset_dir + "illumination/" + ill_str + "/";
		const std::string depth_dir = dataset_dir + "groundtruth/depth_maps/";

		const std::string gt_camera_track_filename = dataset_dir + "groundtruth/camera_track.txt";
		std::ifstream ifs(gt_camera_track_filename);
		if (ifs.fail())
		{
			std::cerr << "Failed to load " + gt_camera_track_filename + "!!!!\n";
			throw std::exception();
		}
		
		std::string line;
		for (int count = 1; std::getline(ifs, line); ++count)
		{
			time_stamps_.push_back(std::to_string(count));
			std::stringstream ss(line);
			// X Y Z A B C : x, y, z, theta_x, theta_y, theta_z
			Vector6f pose_center;
			std::string s;
			for (int i = 0; i < 6; ++i)
			{
				ss >> s;
				if (i >= 3) pose_center(i) = std::stof(s)/ 180.f * M_PI;
				else pose_center(i) = std::stof(s) * rescale_;
			}
			Matrix3f R = Eigen::AngleAxisf(M_PI, Vector3f(0, 1, 0)).toRotationMatrix();
			// Matrix3f R = (Eigen::AngleAxisf(-pose_center(2), Vector3f::UnitZ()) * Eigen::AngleAxisf(pose_center(1), Vector3f::UnitY()) * Eigen::AngleAxisf(pose_center(0), Vector3f::UnitX())).matrix();
			pose_center(2) = - pose_center(2);

			SE3f xi = SE3f(SO3f::exp(pose_center.tail(3)), pose_center.head(3));
			SE3f left = SE3f(R, Vector3f(0, 0, 0));
			// SE3f right = SE3f(R, Vector3f(baseline_ / 2.f, 0, 0));
			gt_poses_.push_back(std::vector<Vector6f>{
				(left * xi).log()
				// SE3f(xi, xi.matrix() * pose_center.head(3) * rescale_).log()
				// SE3f(xi, pose_center.head(3) * rescale_ - offset).log(),
				// SE3f(xi, pose_center.head(3) * rescale_ + offset).log()
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
		}
		ifs.close();

		size_ = gt_poses_.size();
	}

	inline cv::Mat getImage(int index, bool flag_right=false, bool flag_color=false)
	{
		assert(0 <= index && index < size_);
		cv::Mat img = cv::imread(imgs_filenames_[index][flag_right], flag_color);
		return img;
	}

	inline cv::Mat getDepth(int index, bool flag_right=false)
	{
		assert(0 <= index && index < size_);
		cv::FileStorage fs(depths_filenames_[index][flag_right], cv::FileStorage::READ);
		if (!fs.isOpened())
		{
			std::cerr << depths_filenames_[index][flag_right] << " is opened...\n";
			throw std::exception();
		}
		cv::Mat depth;
		fs["depth"] >> depth;
		fs.release();
		depth.convertTo(depth, CV_32F, rescale_);
		return depth;
	}

	inline SE3f getPose(int index, bool flag_right=false)
	{
		assert(0 <= index && index < size_);
		return SE3f::exp(gt_poses_[index][flag_right]);
	}

private:
	const float rescale_ = 0.01f;

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
			std::cerr << "illumination type is wrong!!!\n";
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

	TUMRGBDDataset(const std::string dataset_dir, const TUMRGBD tumrgbd)
	{
		setCameraParameters(tumrgbd);

		std::ifstream ifs(dataset_dir + "associations.txt");
		if (ifs.fail())
		{
			std::cerr << "Failed to load " + dataset_dir + "associations.txt!!!!!!!!\n";
			throw std::exception();
		}
		
		std::string line;
		while (std::getline(ifs, line))
		{
			std::stringstream ss(line);
			Vector7f tmp;
			std::string s;
			for (int i = 0; i < 12; ++i)
			{
				ss >> s;
				if (i == 0)
				{
					time_stamps_.push_back(s);
				}
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
			gt_poses_.push_back(std::vector<Vector6f>{ SE3f(q, tmp.head(3)).log() });
		}
		ifs.close();
		size_ = gt_poses_.size();
	}

	inline cv::Mat getImage(int index, bool flag_rgb=false)
	{
		assert(0 <= index && index < size_);
		cv::Mat img = cv::imread(imgs_filenames_[index][0], flag_rgb);
		return img;
	}

	inline cv::Mat getDepth(int index)
	{
		cv::Mat depth = cv::imread(depths_filenames_[index][0], cv::IMREAD_UNCHANGED);
		depth.convertTo(depth, CV_32F, 1.f / 5000.f);
		return depth;
	}

	inline SE3f getPose(int index)
	{
		assert(0 <= index && index < size_);
		return SE3f::exp(gt_poses_[index][0]);
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
		// living room
		LR_KT0, LR_KT1, LR_KT2, LR_KT3,
		// office room
		OF_KT0, OF_KT1, OF_KT2, OF_KT3
	};

	ICLNUIMDataset(const std::string dataset_dir, const ICLNUIM iclnuim)
	{
		cam_ = std::make_shared<Camera>(640, 480, 481.2f, -480.0f, 319.5f, 239.5f);

		std::ifstream ifs(dataset_dir + getPoseFilename(iclnuim));
		if (ifs.fail())
		{
			std::cerr << "Failed to load " + dataset_dir + getPoseFilename(iclnuim) + "!!!!!\n";
			throw std::exception();
		}
		
		std::string line;
		while (std::getline(ifs, line))
		{
			std::stringstream ss(line);
			Vector7f tmp;
			std::string s;
			for (int i = 0; i < 8; ++i)
			{
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
			gt_poses_.push_back(std::vector<Vector6f>{ SE3f(q, tmp.head(3)).log() });
		}
		ifs.close();
		size_ = gt_poses_.size();
	}
		
	inline cv::Mat getImage(int index, bool flag_rgb=false)
	{
		assert(0 <= index && index < size_);
		cv::Mat img = cv::imread(imgs_filenames_[index][0], flag_rgb);
		return img;
	}

	inline cv::Mat getDepth(int index)
	{
		cv::Mat depth = cv::imread(depths_filenames_[index][0], cv::IMREAD_UNCHANGED);
		depth.convertTo(depth, CV_32F, 1.f / 5000.f);
		return depth;
	}

	inline SE3f getPose(int index)
	{
		assert(0 <= index && index < size_);
		return SE3f::exp(gt_poses_[index][0]);
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
	KITTIDataset(const std::string dataset_dir, const int sequence_index=0)
	{
		assert(sequence_index >= 0 && sequence_index <= 12);
		setCameraParameters(sequence_index);

		const std::string sequence_str = getZeropadStr(sequence_index, 2);
		const std::string gray_dataset_dir = dataset_dir + "data_odometry_gray/" + sequence_str + "/";
		std::ifstream ifs_times(gray_dataset_dir + "times.txt");
		if (ifs_times.fail())
		{
			std::cerr << "Failed to load " + gray_dataset_dir + "times.txt!!!!\n";
			throw std::exception();
		}
		
		std::string line;
		for (int count = 0; std::getline(ifs_times, line); ++count)
		{
			time_stamps_.push_back(line);

			std::string count_zeropad = getZeropadStr(count, 6);
			imgs_filenames_.push_back(std::vector<std::string>{
				gray_dataset_dir + "image_0/" + count_zeropad + ".png", 
				gray_dataset_dir + "image_1/" + count_zeropad + ".png"
			});
		}
		ifs_times.close();
		size_ = time_stamps_.size();

		std::string pose_filename = dataset_dir + "data_odometry_poses/dataset/poses/" + sequence_str + ".txt";
		std::ifstream ifs_poses(pose_filename);
		if (ifs_poses.fail())
		{
			std::cerr << "Failed to load " + pose_filename + "!!!!!\n";
			throw std::exception();
		}

		while (std::getline(ifs_poses, line))
		{
			std::stringstream ss(line);
			std::string s;
			Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
			for (int y = 0; y < 3; ++y)
			{
				for (int x = 0; x < 4; ++x)
				{
					ss >> s;
					std::cout << s << " ";
					T(y, x) = std::stof(s);
				}
			}
			gt_poses_.push_back(std::vector<Vector6f>{ SE3f(T).log(), Vector6f::Zero() });
		}
	}

	inline cv::Mat getImage(int index, bool flag_right=false, bool flag_color=false)
	{
		assert(0 <= index && index < size_);
		cv::Mat img = cv::imread(imgs_filenames_[index][flag_right], flag_color);
		return img;
	}

	inline SE3f getPose(int index, bool flag_right=false)
	{
		assert(0 <= index && index < size_);
		return SE3f::exp(gt_poses_[index][flag_right]);
	}


private:
	void setCameraParameters(const int sequence_index)
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
		// Set camera parameters
		cam_ = std::make_shared<Camera>(458.654, 457.296, 367.215, 248.375);
		cam_->setDistortions(-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05, 0);

		std::string dataset_path = euroc_dataset_dir + getString(euroc) + "/mav0/";

		std::string time_stamps_filename = dataset_path + "cam0/data.csv";
		std::ifstream ifs_time_stamps(time_stamps_filename);
		if (ifs_time_stamps.fail())
		{
			std::cerr << "Failed to load " + time_stamps_filename + "!!!!\n";
			throw std::exception();
		}

		for (std::string line; std::getline(ifs_time_stamps, line);)
		{
			if (line.size() < 20 || line[0] == '#') continue;
			time_stamps_.push_back(line.substr(0, 19));

			std::string img_filename = line.substr(20, line.size() - 21);
			imgs_filenames_.push_back(std::vector<std::string>{
				dataset_path + "cam0/data/" + img_filename,
				dataset_path + "cam1/data/" + img_filename
			});
			gt_poses_.push_back(std::vector<Vector6f>{ Vector6f::Zero(), Vector6f::Zero() });
		}
		size_ = gt_poses_.size();
	}

	inline cv::Mat getImage(int index, bool flag_right=false, bool flag_color=false)
	{
		assert(0 <= index && index < size_);
		cv::Mat img = cv::imread(imgs_filenames_[index][flag_right], flag_color);
		return img;
	}

	inline SE3f getPose(int index, bool flag_right=false)
	{
		assert(0 <= index && index < size_);
		return SE3f::exp(gt_poses_[index][flag_right]);
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
