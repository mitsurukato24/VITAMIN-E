#include "common_types.h"
#include "dataset.h"
#include "system.h"
#include <opencv2/core/eigen.hpp>

using namespace std;


int main(int argc, char **argv)
{
	// string dataset_dir = "C:/Dataset/";
	string dataset_dir = "/media/mitsurukato24/01D718C7481938D0/Dataset/";

	string tsukuba = dataset_dir + "NewTsukubaStereoDataset/";
	// NewStereoTsukubaDataset dataset(tsukuba);

	string rgbd_freiburg1_xyz = dataset_dir + "TUMRGB-D/rgbd_dataset_freiburg1_xyz/";
	// TUMRGBDDataset dataset(rgbd_freiburg1_xyz, TUMRGBDDataset::TUMRGBD::FREIBURG1);

	string icl_nuim_lr_kt2 = dataset_dir + "ICL-NUIM/living_room_traj2_frei_png/";
	// ICLNUIMDataset dataset(icl_nuim_lr_kt2, ICLNUIMDataset::ICLNUIM::LR_KT2);

	// string kitti = "C:/Dataset/KITTI/";
	string kitti = dataset_dir + "KITTI/";
	KITTIDataset dataset(kitti, 3);

	string euroc_dataset_dir = "/home/mitsurukato24/Dataset/EUROC/";
	// EUROCDataset dataset(euroc_dataset_dir, EUROCDataset::EUROC::V1_01);
	
	double sum = 0.;

	auto cam = dataset.getCamera();
	auto system = std::make_shared<System>(cam);

	for (int index = 0; index < dataset.size(); ++index)
	{
		Timer timer("Frame " + std::to_string(index));

		// Read Image
		cv::Mat_<uchar> img = dataset.getImage(index);

		Sophus::SE3d gt_pose = dataset.getPose(index).cast<double>();
		system->setGTPose(gt_pose);

		auto current_frame = Frame::createFrame(img);
		system->addFrame(current_frame);
		
		double process_time = timer.print("Frame - " + std::to_string(index));
		sum += process_time;
		printf("[FPS] : %f\n", ((index + 1.)/ sum) * 1000.f);

		if (index == 0) continue;
		cv::imshow("img", img);
		cv::Mat debug_kappa;
		cv::applyColorMap(current_frame->kappa_, debug_kappa, cv::COLORMAP_JET);  // BRUE -> RED
		cv::imshow("Debug - Curvature", debug_kappa);

		system->debugFeatureMatching();
		system->debugDenseTracking();
		int key = cv::waitKey(1);
		if (key == 's') key = cv::waitKey(0);
		if (key == 'q') break;
	}
	return 0;
}
