#pragma once
#include "common_types.h"
#include "viewer.h"
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>
#include <opengv/sac_problems/relative_pose/EigensolverSacProblem.hpp>
#include <opengv/triangulation/methods.hpp>


enum class SystemStatus { INITING, TRACKING_GOOD, TRACKING_BAD, LOST };


class System
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<System> Ptr;
    System(const Camera::Ptr &cam)
    {
        cam_ = std::move(cam);
        brief_ptr = cv::xfeatures2d::BriefDescriptorExtractor::create(32, true);
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
        map_ = std::make_shared<Map>();
        viewer_ = std::make_shared<Viewer>(cam_);
        viewer_->setMap(map_);
    }

    bool addFrame(const Frame::Ptr &frame)
    {
        Timer timer("Add frame");
        last_frame_ = std::move(current_frame_);
        current_frame_ = std::move(frame);
        timer.printTime("--- swap frame");

        current_frame_->featureExtract(brief_ptr);
        timer.printTime("--- feature extract");
        switch (status_)
        {
        case SystemStatus::INITING:
            status_ = SystemStatus::TRACKING_GOOD;
            current_frame_->T_i_w_ = gt_poses_[0];
            buildInitMap();
            break;
        case SystemStatus::TRACKING_GOOD:
            track();
            break; 
        default:
            break;
        }
        timer.printTime();
        return true; 
    }

    void setGTPose(Sophus::SE3d &pose)
    {
        gt_poses_.push_back(pose);
    }

    void debugFeatureMatching()
    {
        printf("Found %d matches...\n", (int)current_frame_->ds_matches_.size());
        std::vector<cv::DMatch> matches = current_frame_->ds_matches_;
        printf("Found %d matches...\n", (int)matches.size());
        const int num_draw_match = std::min(100, (int)matches.size());
        if (num_draw_match == 0) return;
        std::vector<cv::DMatch> matches_show;
        std::vector<cv::KeyPoint> last_kps, curr_kps;
        matches_show.reserve(num_draw_match);
        last_kps.reserve(num_draw_match);
        curr_kps.reserve(num_draw_match);
        for (int i = 0; i < num_draw_match; ++i)
        {
            last_kps.push_back(last_frame_->ds_kps_[matches[i].queryIdx]);
            curr_kps.push_back(current_frame_->ds_kps_[matches[i].trainIdx]);
            matches_show.push_back(cv::DMatch(i, i, matches[i].distance));
        }
        cv::Mat draw_matches;
        cv::drawMatches(last_frame_->ds_img_, last_kps, current_frame_->ds_img_, curr_kps, matches_show, draw_matches);
        cv::resize(draw_matches, draw_matches, cv::Size(draw_matches.cols * current_frame_->resize_scale_, draw_matches.rows * current_frame_->resize_scale_));
        cv::imshow("Debug - Feature Matches", draw_matches);
    }
    
    void debugDenseTracking()
    {
        if (current_frame_->match_data_.matches_.size() == 0) return;
        cv::Mat draw_flow = current_frame_->img_.clone();
        cv::cvtColor(draw_flow, draw_flow, cv::COLOR_GRAY2BGR);
        cv::cvtColor(draw_flow, draw_flow, cv::COLOR_BGR2HSV);
        for (auto &match : current_frame_->match_data_.matches_)
        {
            cv::Point2f last_pt(last_frame_->features_[match.first]->position_[0], last_frame_->features_[match.first]->position_[1]);
            cv::Point2f curr_pt(current_frame_->features_[match.second]->position_[0], current_frame_->features_[match.second]->position_[1]);
            cv::Point2f p = last_pt - curr_pt;
            int deg = (static_cast<int>(std::atan2(p.y, p.x) / M_PI * 180.) + 180) / 2;
            cv::line(draw_flow, last_pt, curr_pt, cv::Scalar(deg, 255, 255));
        }
        cv::cvtColor(draw_flow, draw_flow, cv::COLOR_HSV2BGR);
        cv::imshow("Debug - Dence Tracking", draw_flow);
    }

private:
    void estimateAffine(Eigen::Matrix<double, 2, 3> &affine)
    {
        // TODO: Undistortion
        Timer time_affine("--- Estimate Affine");

        matcher->match(last_frame_->ds_desc_, current_frame_->ds_desc_, current_frame_->ds_matches_);
        std::sort(current_frame_->ds_matches_.begin(), current_frame_->ds_matches_.end(),
            [](auto const &m1, auto const &m2){ return m1.distance < m2.distance; }
        );

        std::vector<cv::Point2d> last_pts, curr_pts;
        int num_use_affine = std::min(40, (int)current_frame_->ds_matches_.size());
        last_pts.reserve(num_use_affine);
        curr_pts.reserve(num_use_affine);
        for (int i=0; i < num_use_affine; ++i)
        {
            last_pts.push_back(last_frame_->ds_kps_[current_frame_->ds_matches_[i].queryIdx].pt);
            curr_pts.push_back(current_frame_->ds_kps_[current_frame_->ds_matches_[i].trainIdx].pt);
        }

        cv::Mat_<double> affine_mat = cv::estimateAffine2D(last_pts, curr_pts);
        affine << affine_mat(0, 0), affine_mat(0, 1), affine_mat(0, 2) * current_frame_->resize_scale_,
            affine_mat(1, 0), affine_mat(1, 1), affine_mat(1, 2) * current_frame_->resize_scale_;

        time_affine.printTime();
    }

    void denseTracking(const Eigen::Matrix<double, 2, 3> &affine)
    {
        // TODO: Undistortion
        Timer timer("--- Dense Tracking");

        cv::Mat_<uchar> l_kappa = last_frame_->kappa_;
        cv::Mat_<uchar> c_kappa = current_frame_->kappa_;
        std::vector<Eigen::Vector2d> l_features, c_features;
        l_features.assign(l_kappa.cols * l_kappa.rows, Eigen::Vector2d(-1, -1));  // for thread safe, store at [col][row]
        c_features.assign(c_kappa.cols * c_kappa.rows, Eigen::Vector2d(-1, -1));
        last_frame_->kappa_.forEach(
            [&l_kappa, &c_kappa, &affine, &l_features, &c_features](uchar &pixel, const int* position) -> void
            {
                const double roi = 10.f;
                const double lambda = 5.f;
                const double sigma2 = 5.f;
                const double step_size = 0.2f;
                const int height = l_kappa.rows;
                const int width = l_kappa.cols;
                const int x = position[1], y = position[0];
                const uchar threshold = 50;
                
                Eigen::Vector2d last_pt(x, y);
                // get initial guess
                Eigen::Vector2d init_pt = affine * Eigen::Vector3d(x, y, 1.f);
                bool flag_invalid = init_pt[0] < roi || init_pt[0] > width - roi || init_pt[1] < roi || init_pt[1] > height - roi;
                if (l_kappa.ptr(y)[x] < threshold || flag_invalid) return;
                double init_f = getSubpixelValue(c_kappa.data, init_pt[0], init_pt[1], width, height)/255.f + lambda;

                double max_f = init_f;
                Eigen::Vector2d max_pt = init_pt;
                for (int i = 0; i < 50; ++i)
                {
                    bool flag_found = false;
                    for (int dy = -1; dy <= 1; ++dy)
                    {
                        for (int dx = -1; dx <= 1; ++dx)
                        {
                            if (dx == 0 && dy == 0) continue;
                            Eigen::Vector2d tmp_pt(max_pt[0] + dx * step_size, max_pt[1] + dy * step_size);
                            if (tmp_pt[0] < 0.f || tmp_pt[0] > width - 1.f || tmp_pt[1] < 0.f || tmp_pt[1] > height - 1.f) return;

                            double tmp_f = getSubpixelValue(c_kappa.data, tmp_pt[0], tmp_pt[1], width, height) / 255.f
                                + lambda * sigma2 / ((max_pt - tmp_pt).squaredNorm() + sigma2);
                            if (max_f < tmp_f)
                            {
                                max_f = tmp_f;
                                max_pt = tmp_pt;
                                flag_found = true;
                            }
                        }
                    }
                    if (!flag_found) break;
                }
                l_features[y * width + x] = last_pt;
                c_features[y * width + x] = max_pt;
            }
        );
        timer.printTime("tracking");

        l_features.erase(std::remove(l_features.begin(), l_features.end(), Eigen::Vector2d(-1, -1)));
        c_features.erase(std::remove(c_features.begin(), c_features.end(), Eigen::Vector2d(-1, -1)));
        timer.printTime("removing");

        uint l_size = last_frame_->features_.size();


        current_frame_->features_.reserve(c_kappa.cols * c_kappa.rows);
        current_frame_->match_data_.matches_.reserve(c_kappa.cols * c_kappa.rows);
        for (size_t i = 0, c = 0; i < c_features.size(); ++i)
        {
            if (c_features[i][0] < 0) continue;
            last_frame_->features_.emplace_back(std::make_shared<Feature>(l_features[i]));
            current_frame_->features_.emplace_back(std::make_shared<Feature>(c_features[i]));
            current_frame_->match_data_.matches_.emplace_back(std::pair<uint, uint>(l_size + c, c));
            ++c;
        }
        timer.printTime("push");

        timer.printTime();
    }

    inline static bool estimateRotation(
        const Frame::Ptr &frame0, const Frame::Ptr &frame1, const Camera::Ptr &cam,
        const double ransac_reproj_error_inlier_threshld_pixel, const int num_ransac_min_inliers, 
        Sophus::SO3d &R_1_0
    )
    {
        Timer timer("--- Estimate Rotation");

        using namespace opengv;
        if ((int)frame1->match_data_.matches_.size() < num_ransac_min_inliers) return false;

        bearingVectors_t bvs0, bvs1;
        bvs0.reserve(frame1->match_data_.matches_.size());
        bvs1.reserve(frame1->match_data_.matches_.size());
        for (size_t i = 0; i < frame1->match_data_.matches_.size(); ++i)
        {
            // faster than reserve & push_back?
            const auto m = frame1->match_data_.matches_[i];
            bvs0.emplace_back(bearingVector_t(cam->unproject(frame0->features_[m.first]->position_)));
            bvs1.emplace_back(bearingVector_t(cam->unproject(frame1->features_[m.second]->position_)));
        }
        timer.printTime("--- make bearing vectors");
        
        relative_pose::CentralRelativeAdapter adapter(bvs0, bvs1);
        sac::Ransac<sac_problems::relative_pose::EigensolverSacProblem> ransac;
        std::shared_ptr<sac_problems::relative_pose::EigensolverSacProblem>
            eigenproblem_ptr(new sac_problems::relative_pose::EigensolverSacProblem(adapter, 10));
        ransac.sac_model_ = eigenproblem_ptr;
        ransac.threshold_ = 1. - cos(atan(sqrt(ransac_reproj_error_inlier_threshld_pixel) * 0.5 / cam->fx_));
        ransac.max_iterations_ = 30;

        bool success = ransac.computeModel();

        if (!success || static_cast<int>(ransac.inliers_.size()) < num_ransac_min_inliers)
        {
            printf("insufficient inliers!!! (%d)\n", (int)ransac.inliers_.size());
            return false;
        }
        
        sac_problems::relative_pose::EigensolverSacProblem::model_t optimized_model;
        size_t prev_num_inliers;
        do
        {
            prev_num_inliers = ransac.inliers_.size();
            ransac.sac_model_->optimizeModelCoefficients(
                ransac.inliers_, ransac.model_coefficients_, optimized_model);
            ransac.sac_model_->selectWithinDistance(
                optimized_model, ransac.threshold_, ransac.inliers_);
        }
        while (ransac.inliers_.size() > prev_num_inliers);

        if (static_cast<int>(ransac.inliers_.size()) < num_ransac_min_inliers)
        {
            printf("insufficient inliers!!! (%d)\n", (int)ransac.inliers_.size());
            return false;
        }

        

        frame1->match_data_.inliers_.clear();
        frame1->match_data_.inliers_.reserve(ransac.inliers_.size());
        for (size_t i = 0; i < ransac.inliers_.size(); ++i)
        {
            frame1->match_data_.inliers_.push_back(ransac.inliers_[i]);
        }
        R_1_0 = Sophus::SO3d(optimized_model.rotation);
        
        timer.printTime();
        return true;
    }

    inline static bool estimateTranslation(
        const Frame::Ptr &frame0, const Frame::Ptr &frame1, 
        const Sophus::SO3d &R_1_0, const Camera::Ptr &cam,
        const double ransac_reproj_error_inlier_threshld_pixel, const int num_ransac_min_inliers,
        Sophus::SE3d &T_1_0
    )
    {
        Timer timer("--- Estimate Translation");

        using namespace opengv;
        bearingVectors_t bvs0(frame1->match_data_.inliers_.size()), bvs1(frame1->match_data_.inliers_.size());
        for (size_t i = 0; i < frame1->match_data_.inliers_.size(); ++i)
        {
            const int inlier = frame1->match_data_.inliers_[i];
            bvs0[i] = bearingVector_t(cam->unproject(frame0->features_[frame1->match_data_.matches_[inlier].first]->position_).normalized());
            bvs1[i] = bearingVector_t(cam->unproject(frame1->features_[frame1->match_data_.matches_[inlier].second]->position_).normalized());
        }
        timer.printTime("--- push bearing vector");

        bool flag_5pt = true;
        if (flag_5pt)
        {
            relative_pose::CentralRelativeAdapter adapter(bvs0, bvs1, R_1_0.matrix());
            sac::Ransac<sac_problems::relative_pose::CentralRelativePoseSacProblem> ransac;
            std::shared_ptr<sac_problems::relative_pose::CentralRelativePoseSacProblem>
                problem_ptr(new sac_problems::relative_pose::CentralRelativePoseSacProblem(adapter, sac_problems::relative_pose::CentralRelativePoseSacProblem::STEWENIUS));
            ransac.sac_model_ = problem_ptr;
            ransac.threshold_ = 1. - cos(atan(sqrt(ransac_reproj_error_inlier_threshld_pixel) * 0.5 / cam->fx_));
            ransac.max_iterations_ = 30;

            bool success = ransac.computeModel();

            if (!success || static_cast<int>(ransac.inliers_.size()) < num_ransac_min_inliers)
            {
                printf("insufficient inliers!!! (%d)\n", (int)ransac.inliers_.size());
                return false;
            }
            
            transformation_t optimized_pose = ransac.model_coefficients_;
            size_t prev_num_inliers;
            do
            {
                prev_num_inliers = ransac.inliers_.size();
                ransac.sac_model_->optimizeModelCoefficients(
                    ransac.inliers_, ransac.model_coefficients_, optimized_pose);
                ransac.sac_model_->selectWithinDistance(
                    optimized_pose, ransac.threshold_, ransac.inliers_);
            }
            while (ransac.inliers_.size() > prev_num_inliers);

            if (static_cast<int>(ransac.inliers_.size()) < num_ransac_min_inliers)
            {
                printf("insufficient inliers!!! (%d)\n", (int)ransac.inliers_.size());
                return false;
            }
            else{printf("%d inliers :)\n", (int)ransac.inliers_.size());}

            frame1->match_data_.inliers_.clear();
            frame1->match_data_.inliers_.reserve(ransac.inliers_.size());
            for (size_t i = 0; i < ransac.inliers_.size(); ++i)
            {
                frame1->match_data_.inliers_[i] = ransac.inliers_[i];
            }
            T_1_0 = Sophus::SE3d(optimized_pose.leftCols<3>(), optimized_pose.rightCols<1>());
        }
        else
        {
            relative_pose::CentralRelativeAdapter adapter(bvs0, bvs1, R_1_0.matrix());
            size_t iterations = 5;

            //running experiments
            translation_t twopt_translation = relative_pose::twopt(adapter, true);
            adapter.sett12(twopt_translation.rightCols<1>());
            transformation_t nonlinear_transformation;
            for(size_t i = 0; i < iterations; i++)
            {
                nonlinear_transformation = relative_pose::optimize_nonlinear(adapter);
            }

            T_1_0 = Sophus::SE3d(nonlinear_transformation.leftCols<3>(), nonlinear_transformation.rightCols<1>());
        }
        timer.printTime();
        return true;
    }

    double calcMeanDisparity(const Frame::Ptr &frame0, const Frame::Ptr &frame1, const Sophus::SO3d &R_1_0)
    {
        printf("%d inliers\n", (int)frame1->match_data_.inliers_.size());
        double sum_disp = 0.;
        for (auto &inlier : frame1->match_data_.inliers_)
        {
            sum_disp += (
                frame1->features_[frame1->match_data_.matches_[inlier].second]->position_
                    - cam_->project(R_1_0 * cam_->unproject(frame0->features_[frame1->match_data_.matches_[inlier].first]->position_))
            ).norm();
        }
        return sum_disp / (double)frame1->match_data_.inliers_.size();
    }

    bool P3P(Frame::Ptr frame0, Frame::Ptr frame1)
    {
        return true;
    }

    inline int triangulate(const Frame::Ptr &frame0, const Frame::Ptr &frame1, const Sophus::SE3d &T_1_0)
    {
        Timer timer("Triangulate");

        using namespace opengv;
        bearingVectors_t bvs0, bvs1;
        bvs0.reserve(frame1->match_data_.inliers_.size());
        bvs1.reserve(frame1->match_data_.inliers_.size());
        for (size_t i = 0; i < frame1->match_data_.inliers_.size(); ++i)
        {
            // faster than reserve & push_back?
            const auto m = frame1->match_data_.matches_[frame1->match_data_.inliers_[i]];
            bvs0.emplace_back(bearingVector_t(cam_->unproject(frame0->features_[m.first]->position_)));
            bvs1.emplace_back(bearingVector_t(cam_->unproject(frame1->features_[m.second]->position_)));
        }
        timer.printTime("--- make bearing vectors");

        // Set adapter
        relative_pose::CentralRelativeAdapter adapter(
            bvs0, bvs1, T_1_0.translation(), T_1_0.rotationMatrix());

        int count = 0;
        const size_t iterations = 100;
        for (size_t j = 0; j < frame1->match_data_.inliers_.size(); ++j)
        {
            // Triangulate
            for (size_t i = 0; i < iterations - 1; ++i)
            {
                triangulation::triangulate2(adapter, j);
            }
            if (triangulation::triangulate2(adapter, j).z() < 0)
            {
                continue;
            }

            auto new_landmark = Landmark::createNewLandmark();
            new_landmark->position_ = frame0->T_i_w_.rotationMatrix() * triangulation::triangulate2(adapter, j) + frame0->T_i_w_.translation();
            const auto m = frame1->match_data_.matches_[frame1->match_data_.inliers_[j]];
            new_landmark->addObservation(frame0->features_[m.first]);
            new_landmark->addObservation(frame1->features_[m.second]);
            frame0->features_[m.first]->landmark_ = new_landmark;
            frame1->features_[m.second]->landmark_ = new_landmark;
            map_->insertLandmark(new_landmark);
            ++count;
        }

        timer.printTime();
        return count;
    }

    bool initializePose()
    {
        const double ransac_reproj_error_inlier_threshld_pixel = 2.;
        const int num_ransac_min_inlier = 20;

        // Estimate relative rotation
        Sophus::SO3d relative_R;
        bool success_r = estimateRotation(last_frame_, current_frame_, cam_, ransac_reproj_error_inlier_threshld_pixel, num_ransac_min_inlier, relative_R);
        if (!success_r) return false;

        // Calculation mean disparity
        double mean_disp = calcMeanDisparity(last_frame_, current_frame_, relative_R);
        printf("------------------ Mean Disparity : %f\n", mean_disp);
        double disp_threshold = 0.5;
        if (mean_disp < disp_threshold) return false;
        // return false;

        // Estimate relative translation
        Sophus::SE3d T_c_l;
        /*
        bool success_t = estimateTranslation(
            last_frame_, current_frame_, relative_R, cam_,
            ransac_reproj_error_inlier_threshld_pixel, num_ransac_min_inlier, T_c_l
        );
        current_frame_->setPose(T_c_l * last_frame_->getPose());
        if (!success_t) return false;
        */
        T_c_l = last_frame_->T_i_w_.inverse() * current_frame_->T_i_w_;

        triangulate(last_frame_, current_frame_, T_c_l);

        return true;
    }

    bool track()
    {
        Eigen::Matrix<double, 2, 3> affine;
        estimateAffine(affine);
        denseTracking(affine);

        current_frame_->setPose(gt_poses_[gt_poses_.size()-1]);
        
        bool success = initializePose();
        

        // trianguration for p3p and initial guess for BA

        insertKeyframe();

        // p3p

        if (viewer_) viewer_->addCurrentFrame(current_frame_);

        return true;
    }
    
    bool insertKeyframe()
    {
        // use all frame as keyframe ... 
        current_frame_->setKeyFrame();
        map_->insertKeyFrame(current_frame_);

        setObservationsForKeyframe();
        // triangulate

        if (viewer_) viewer_->updateMap();
        return true;
    }

    void setObservationsForKeyframe()
    {
        for (auto &inlier : current_frame_->match_data_.inliers_)
        {
            auto m = current_frame_->match_data_.matches_[inlier];
            auto landmark = current_frame_->features_[m.second]->landmark_.lock();
            if (landmark) landmark->addObservation(current_frame_->features_[m.second]);
        }
    }

    bool buildInitMap()
    {
        current_frame_->setKeyFrame();
        map_->insertKeyFrame(current_frame_);
        return true;
    }

    SystemStatus status_ = SystemStatus::INITING;
    Frame::Ptr current_frame_ = nullptr;
    Frame::Ptr last_frame_ = nullptr;
    Camera::Ptr cam_ = nullptr;
    FramesType frames_;
    Map::Ptr map_ = nullptr;
    Viewer::Ptr viewer_ = nullptr;

    std::vector<Sophus::SE3d> gt_poses_;

    cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> brief_ptr;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    using FrameIdPair = std::pair<unsigned int, unsigned int>;
    // std::unordered_map<FrameIdPair, MatchData, pair_hash<FrameIdPair>, std::equal_to<FrameIdPair>, Eigen::aligned_allocator<std::pair<const FrameIdPair, MatchData>>> match_data_;
};
