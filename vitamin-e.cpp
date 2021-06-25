#include "common_types.h"
#include "dataset.h"
#include "system.h"

using namespace std;


int main(int argc, char **argv)
{
	// string dataset_dir = "C:/Dataset/";
	string dataset_dir = "/media/mitsurukato24/01D718C7481938D0/Dataset/";

	string icl_nuim_lr_kt2 = dataset_dir + "ICL-NUIM/living_room_traj2_frei_png/";
	// ICLNUIMDataset dataset(icl_nuim_lr_kt2, ICLNUIMDataset::ICLNUIM::LR_KT2);

	string rgbd_freiburg1_xyz = dataset_dir + "TUMRGB-D/rgbd_dataset_freiburg1_xyz/";
	// TUMRGBDDataset dataset(rgbd_freiburg1_xyz, TUMRGBDDataset::TUMRGBD::FREIBURG1);
	
	string tsukuba = dataset_dir + "NewTsukubaStereoDataset/";
	NewStereoTsukubaDataset dataset(tsukuba);

	// string kitti = "C:/Dataset/KITTI/data_odometry_gray/";
	string kitti = dataset_dir + "KITTI/data_odometry_gray/";
	// KITTIDataset dataset(kitti, 2);

	bool flag_initialize = true;
	double sum = 0.;
	MeasureTime measure_time;

	auto system = std::make_shared<System>();

	std::vector<Frame::Ptr> frames;
	frames.reserve(dataset.size());
	Frame::Ptr current_frame, last_frame;
	for (int index = 0; index < dataset.size(); ++index)
	{
		measure_time.start();

		// Read Image
		cv::Mat img;
		dataset.getData(index, img, false);
		current_frame = Frame::CreateFrame(img);
		system->addFrame(current_frame);
		
		double process_time = measure_time.printTime("Total");
		sum += process_time;
		printf("[FPS] : %f\n", ((index + 1.)/ sum) * 1000.f);

		if (index == 0) continue;
		cv::imshow("img", img);
		cv::Mat debug_kappa;
		cv::applyColorMap(current_frame->kappa_, debug_kappa, cv::COLORMAP_JET);  // BRUE -> RED
		cv::imshow("Debug - Curvature", debug_kappa);

		// system->debugFeatureMatching();
		system->debugDenseTracking();
		int key = cv::waitKey(1);
		if (key == 's')
		{
			key = cv::waitKey(0);
		}
		if (key == 'q') break;
	}
	return 0;
}
