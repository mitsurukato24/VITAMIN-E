#pragma once
#include "common_types.h"

typedef std::unordered_map<unsigned long, Frame::Ptr> Frames;


enum class SystemStatus { INITING, TRACKING_GOOD, TRACKING_BAD, LOST };


class System
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<System> Ptr;
    System()
    {
        brief_ptr = cv::xfeatures2d::BriefDescriptorExtractor::create(32, true);
        matcher = cv::BFMatcher::create(cv::BFMatcher::BRUTEFORCE_HAMMINGLUT, true);
    }

    bool addFrame(Frame::Ptr frame)
    {
        MeasureTime swap_time;
        last_frame_ = std::move(current_frame_);
        swap_time.printTime("---------- Swap Frame");

        current_frame_ = frame;
        current_frame_->featureExtract(brief_ptr);
        switch (status_)
        {
        case SystemStatus::INITING:
            status_ = SystemStatus::TRACKING_GOOD;
            break;
        case SystemStatus::TRACKING_GOOD:
            track();
            break; 
        default:
            break;
        }
        return true; 
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
        cv::resize(draw_matches, draw_matches, cv::Size(draw_matches.cols * 4, draw_matches.rows * 4));
        cv::imshow("Debug - Feature Matches", draw_matches);
    }
    
    void estimateAffine(Eigen::Matrix<float, 2, 3> &affine)
    {
        MeasureTime time_affine;

        matcher->match(last_frame_->ds_desc_, current_frame_->ds_desc_, current_frame_->ds_matches_);
        std::sort(current_frame_->ds_matches_.begin(), current_frame_->ds_matches_.end(),
            [](auto const &m1, auto const &m2){ return m1.distance < m2.distance; }
        );

        std::vector<cv::Point2d> last_pts, curr_pts;
        int num_use_affine = std::min(100, (int)current_frame_->ds_matches_.size());
        last_pts.reserve(num_use_affine);
        curr_pts.reserve(num_use_affine);
        for (int i=0; i < num_use_affine; ++i)
        {
            last_pts.push_back(last_frame_->ds_kps_[current_frame_->ds_matches_[i].queryIdx].pt);
            curr_pts.push_back(current_frame_->ds_kps_[current_frame_->ds_matches_[i].trainIdx].pt);
        }
        cv::Mat_<float> affine_mat = cv::estimateAffine2D(last_pts, curr_pts);
        affine << affine_mat(0, 0), affine_mat(0, 1), affine_mat(0, 2), affine_mat(1, 0), affine_mat(1, 1), affine_mat(1, 2);

        time_affine.printTime("---------- Estimate Affine");
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
            uchar v = cv::norm(p);
            cv::line(draw_flow, last_pt, curr_pt, cv::Scalar(deg, 255, 255));
        }
        cv::cvtColor(draw_flow, draw_flow, cv::COLOR_HSV2BGR);
        cv::imshow("Debug - Dence Tracking", draw_flow);
    }

    void denseTracking(const Eigen::Matrix<float, 2, 3> &affine)
    {
        MeasureTime measure_time;

        cv::Mat l_kappa = last_frame_->kappa_;
        cv::Mat c_kappa = current_frame_->kappa_;
        std::vector<Eigen::Vector2f> l_features, c_features;
        l_features.assign(l_kappa.cols * l_kappa.rows, Eigen::Vector2f(-1, -1));  // for thread safe, store at [col][row]
        c_features.assign(c_kappa.cols * c_kappa.rows, Eigen::Vector2f(-1, -1));
        last_frame_->kappa_.forEach<uchar>(
            [&l_kappa, &c_kappa, &affine, &l_features, &c_features](uchar &pixel, const int* position) -> void
            {
                const float roi = 10.f;
                const float lambda = 5.f;
                const float sigma2 = 5.f;
                const float step_size = 0.5f;
                const int height = l_kappa.rows;
                const int width = l_kappa.cols;
                const int x = position[1], y = position[0];
                const uchar threshold = 50;
                int feature_id = 0;
                
                Eigen::Vector2f last_pt(x, y);
                // get initial guess
                Eigen::Vector2f init_pt = affine * Eigen::Vector3f(x, y, 4.f);
                bool flag_invalid = init_pt[0] < roi || init_pt[0] > width - roi || init_pt[1] < roi || init_pt[1] > height - roi;
                if (l_kappa.at<uchar>(y, x) < threshold || flag_invalid) return;
                float init_f = getSubpixelValue((uchar*)c_kappa.data, init_pt[0], init_pt[1], width, height)/255.f + lambda;

                float max_f = init_f;
                Eigen::Vector2f max_pt = init_pt;
                for (int i = 0; i < 30; ++i)
                {
                    bool flag_found = false;
                    for (int dy = -1; dy <= 1; ++dy)
                    {
                        for (int dx = -1; dx <= 1; ++dx)
                        {
                            if (dx == 0 && dy == 0) continue;
                            Eigen::Vector2f tmp_pt(max_pt[0] + dx * step_size, max_pt[1] + dy * step_size);
                            if (tmp_pt[0] < 0.f || tmp_pt[0] > width - 1.f || tmp_pt[1] < 0.f || tmp_pt[1] > height - 1.f) continue;
                            float tmp_f = getSubpixelValue((uchar*)c_kappa.data, tmp_pt[0], tmp_pt[1], width, height) / 255.f
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
        measure_time.printTime("----- Tracking");

        l_features.erase(std::remove(l_features.begin(), l_features.end(), Eigen::Vector2f(-1, -1)));
        c_features.erase(std::remove(c_features.begin(), c_features.end(), Eigen::Vector2f(-1, -1)));
        measure_time.printTime("----- Removing");

        uint l_size = last_frame_->features_.size();
        current_frame_->features_.reserve(c_kappa.cols * c_kappa.rows);
        current_frame_->match_data_.matches_.reserve(c_kappa.cols * c_kappa.rows);
        
        for (int i = 0, c = 0; i < l_features.size(); ++i)
        {
            if (l_features[i][0] < 0) continue;
            // last_frame_->features_.emplace_back(Feature::Ptr(new Feature(last_frame_, l_features[i])));
            // current_frame_->features_.emplace_back(Feature::Ptr(new Feature(current_frame_, c_features[i])));
            // last_frame_->features_.emplace_back(std::make_shared<Feature>(last_frame_, l_features[i]));
            // current_frame_->features_.emplace_back(std::make_shared<Feature>(current_frame_, c_features[i]));
            last_frame_->features_.emplace_back(std::make_shared<Feature>(l_features[i]));
            current_frame_->features_.emplace_back(std::make_shared<Feature>(c_features[i]));
            current_frame_->match_data_.matches_.emplace_back(std::pair<uint, uint>(l_size + c, c));
            ++c;
        }
        measure_time.printTime("----- Push");
        measure_time.printTime("Dense Tracking");
    }

    bool track()
    {
        Eigen::Matrix<float, 2, 3> affine;
        estimateAffine(affine);
        denseTracking(affine);
        return true;
    }

private:
    SystemStatus status_ = SystemStatus::INITING;
    Frame::Ptr current_frame_ = nullptr;
    Frame::Ptr last_frame_ = nullptr;
    Frames frames_;
    cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> brief_ptr;
    cv::Ptr<cv::BFMatcher> matcher;
    // std::unordered_map<std:pair<unsigned int, unsigned int>, std::vector<cv::DMatch>> matches_;
};
