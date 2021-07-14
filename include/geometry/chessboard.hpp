#pragma once

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <vector>

namespace bxg {

class ChessBoard {
public:
    using Vec2 = Eigen::Vector2d;
    static constexpr uint8_t color_black = 50;
    static constexpr uint8_t color_white = 200;

    ChessBoard(int _rows = 7, int _cols = 10, double _step = 35.0)
        : rows(_rows)
        , cols(_cols)
        , step(_step)
    {
        options.x_shift = options.y_shift = -step / 2.0;
        initDiscreteImage();
    }

    /// \brief Initialize and return the discrete board image
    /// \param resolution_rate Num of pixels per millimeter
    /// \param gaussian_ksize The kernel size of gaussian blur (0 means no blur)
    /// \return The generated discrete board image
    cv::Mat initDiscreteImage(int gaussian_ksize = 9)
    {
        const double pix_step = step * options.resolution_rate;
        const int rs = cvRound((rows + 1) * pix_step);
        const int cs = cvRound((cols + 1) * pix_step);
        cv::Mat res(rs, cs, CV_8UC1);
        for (int y = 0; y < rs; ++y) {
            auto ptr = res.ptr<uchar>(y);
            for (int x = 0; x < cs; ++x) {
                int col = cvFloor(x / pix_step);
                int row = cvFloor(y / pix_step);
                ptr[x] = (((row + col) & 1) ^ options.start_with_white) ? color_white : color_black;
            }
        }

        if (gaussian_ksize > 0)
            cv::GaussianBlur(res, res, { gaussian_ksize, gaussian_ksize }, -1);

        image_ = res.clone();
        return res;
    }

    double pixValInterpolated(double x, double y) const
    {
        // minus 0.5 pixel to shift the coordinate from pixel centers to pixel borders
        x = (x + options.x_shift + step - 0.5) * options.resolution_rate;
        y = (y + options.y_shift + step - 0.5) * options.resolution_rate;
        if (x < 0 || x > image_.cols - 1 || y < 0 || y > image_.rows - 1)
            return -1.0;

        const int stride = image_.step1();
        int xl = cvFloor(x), yl = cvFloor(y);
        double dx = x - xl, dy = y - yl;
        const uchar* data = image_.ptr<uchar>(yl) + xl;
        return data[0] * (1.0 - dx) * (1.0 - dy)
            + data[1] * dx * (1.0 - dy)
            + data[stride] * (1.0 - dx) * dy
            + data[stride + 1] * dx * dy;
    }

    uint8_t pixVal(double x, double y) const
    {
        x = x * options.scale + options.x_shift;
        y = y * options.scale + options.y_shift;

        int col = cvFloor(x / step) + 1;
        int row = cvFloor(y / step) + 1;
        if (col < 0 || col > cols
            || row < 0 || row > rows)
            return options.back_ground;
        else
            return (((row + col) & 1) ^ options.start_with_white) ? color_white : color_black;
    }

    void draw(cv::Mat& image) const
    {
        if (image.empty())
            image = cv::Mat((rows + 1) * step, (cols + 1) * step, CV_8UC1);
        CV_Assert(image.type() == CV_8UC1);
        for (int y = 0; y < image.rows; ++y) {
            auto ptr = image.ptr<uchar>(y);
            for (int x = 0; x < image.cols; ++x) {
                ptr[x] = pixVal(x, y);
            }
        }
    }

    std::vector<Vec2> corners() const
    {
        std::vector<Vec2> ret;
        ret.reserve(rows * cols);
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                ret.emplace_back(c * step - options.x_shift, r * step - options.y_shift);
        return ret;
    }

    std::vector<Vec2> squareCenters(std::vector<bool>* colors = nullptr) const
    {
        std::vector<Vec2> ret;
        ret.resize((rows + 1) * (cols + 1));
        if (colors)
            colors->resize(ret.size());
        for (int r = 0; r <= rows; ++r)
            for (int c = 0; c <= cols; ++c) {
                int idx = r * (cols + 1) + c;
                ret[idx] = Vec2((c - 0.5) * step - options.x_shift, (r - 0.5) * step - options.y_shift);
                if (colors)
                    colors->at(idx) = ((r + c) & 1) ^ options.start_with_white;
            }
        return ret;
    }

public:
    struct PixelOptions {
        double x_shift { 0.0 }, y_shift { 0.0 };
        double scale { 1.0 };
        bool start_with_white { true };
        uint8_t back_ground { 126 };

        const double resolution_rate = 1.0;
    };

public:
    const int rows, cols; //!< num of corners
    const double step;

    PixelOptions options;

private:
    cv::Mat image_;
};

} //namespace bxg
