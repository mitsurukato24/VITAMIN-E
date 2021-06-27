#include "common_types.h"
#include <pangolin/pangolin.h>


class Viewer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Viewer> Ptr;

    Viewer(const Camera::Ptr &cam)
    {
        cam_ = std::move(cam);
        viewer_thread_ = std::thread(std::bind(&Viewer::threadLoop, this));
    }

    void setMap(Map::Ptr map)
    {
        map_ = std::move(map);
    }

    void close()
    {
        viewer_running_ = false;
        viewer_thread_.join();
    }

    void addCurrentFrame(Frame::Ptr current_frame)
    {
        std::unique_lock<std::mutex> lock(viewer_data_mutex_);
        current_frame_ = std::move(current_frame);
    }

    void updateMap()
    {
        std::unique_lock<std::mutex> lock(viewer_data_mutex_);
        assert(map_ != nullptr);
        active_keyframes_ = map_->getActiveKeyframes();
        active_landmarks_ = map_->getActiveLandmarks();
        map_updated_ = true;
    }

private:
    void threadLoop()
    {
        pangolin::CreateWindowAndBind("VITAMIN-E", cam_->width_, cam_->height_);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        pangolin::OpenGlRenderState vis_cam(
            pangolin::ProjectionMatrix(cam_->width_, cam_->height_, cam_->fx_, cam_->fy_, cam_->cx_, cam_->cy_, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -5, -10, 0, 0, 0, 0., -1., 0.)
        );

        pangolin::View &vis_display = pangolin::CreateDisplay().SetBounds(
            0., 1., 0., 1., - (float)cam_->width_ / cam_->height_).SetHandler(new pangolin::Handler3D(vis_cam));

        const float blue[3] = {0, 0, 1};
        const float green[3] = {0, 1, 0};

        while(!pangolin::ShouldQuit() && viewer_running_)
        {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glClearColor(1.f, 1.f, 1.f, 1.f);
            vis_display.Activate(vis_cam);

            {
                std::unique_lock<std::mutex> lock(viewer_data_mutex_);
                if (current_frame_)
                {
                    drawFrame(current_frame_, green);
                    followCurrentFrame(vis_cam);

                    // cv::Mat img = plotFrameImage();
                    // cv::imshow("image", img);
                    // cv::waitKey(1);
                }
            }

            if (map_) drawLandmarks();

            pangolin::FinishFrame();
        }

        printf("----Stop Viewer----\n");
    }

    void drawFrame(Frame::Ptr frame, const float* color)
    {
        const float sz = 1.0;
        const int line_width = 2.0;

        glPushMatrix();

        Sophus::Matrix4f m = frame->getPose().matrix().cast<float>();
        glMultMatrixf((GLfloat*)m.data());

        if (color == nullptr) glColor3f(1, 0, 0);
        else glColor3f(color[0], color[1], color[2]);

		glLineWidth(line_width);
		glBegin(GL_LINES);
		glVertex3f(0, 0, 0);
		glVertex3f(sz * (0 - cam_->cx_) / cam_->fx_, sz * (0 - cam_->cy_) / cam_->fy_, sz);
		glVertex3f(0, 0, 0);
		glVertex3f(sz * (0 - cam_->cx_) / cam_->fx_, sz * (cam_->height_ - 1 - cam_->cy_) / cam_->fy_, sz);
		glVertex3f(0, 0, 0);
		glVertex3f(sz * (cam_->width_ - 1 - cam_->cx_) / cam_->fx_, sz * (cam_->height_ - 1 - cam_->cy_) / cam_->fy_, sz);
		glVertex3f(0, 0, 0);
		glVertex3f(sz * (cam_->width_ - 1 - cam_->cx_) / cam_->fx_, sz * (0 - cam_->cy_) / cam_->fy_, sz);

		glVertex3f(sz * (cam_->width_ - 1 - cam_->cx_) / cam_->fx_, sz * (0 - cam_->cy_) / cam_->fy_, sz);
		glVertex3f(sz * (cam_->width_ - 1 - cam_->cx_) / cam_->fx_, sz * (cam_->height_ - 1 - cam_->cy_) / cam_->fy_, sz);

		glVertex3f(sz * (cam_->width_ - 1 - cam_->cx_) / cam_->fx_, sz * (cam_->height_ - 1 - cam_->cy_) / cam_->fy_, sz);
		glVertex3f(sz * (0 - cam_->cx_) / cam_->fx_, sz * (cam_->height_ - 1 - cam_->cy_) / cam_->fy_, sz);

		glVertex3f(sz * (0 - cam_->cx_) / cam_->fx_, sz * (cam_->height_ - 1 - cam_->cy_) / cam_->fy_, sz);
		glVertex3f(sz * (0 - cam_->cx_) / cam_->fx_, sz * (0 - cam_->cy_) / cam_->fy_, sz);

		glVertex3f(sz * (0 - cam_->cx_) / cam_->fx_, sz * (0 - cam_->cy_) / cam_->fy_, sz);
		glVertex3f(sz * (cam_->width_ - 1 - cam_->cx_) / cam_->fx_, sz * (0 - cam_->cy_) / cam_->fy_, sz);

		glEnd();
		glPopMatrix();
    }

    void drawLandmarks()
    {
        const float red[3] = {1, 0, 0};
        for (auto &kf : active_keyframes_)
        {
            drawFrame(kf.second, red);
        }

        glPointSize(2);
        glBegin(GL_POINTS);
        for (auto& landmark : active_landmarks_)
        {
            auto pos = landmark.second->position_;
            glColor3f(red[0], red[1], red[2]);
            glVertex3d(pos[0], pos[1], pos[2]);
        }
        glEnd();

    }

    void followCurrentFrame(pangolin::OpenGlRenderState& vis_cam)
    {
        pangolin::OpenGlMatrix m(current_frame_->getPose().matrix());
        vis_cam.Follow(m, true);
    }

    Frame::Ptr current_frame_ = nullptr;
    Map::Ptr map_ = nullptr;
    Camera::Ptr cam_ = nullptr;

    std::thread viewer_thread_;
    bool viewer_running_ = true;

    std::unordered_map<unsigned int, Frame::Ptr> active_keyframes_;
    std::unordered_map<unsigned int, Landmark::Ptr> active_landmarks_;
    bool map_updated_ = false;

    std::mutex viewer_data_mutex_;
};