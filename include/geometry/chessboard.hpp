#pragma once

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <vector>

namespace bxg {

class ChessBoard {
public:
    using Vec2 = Eigen::Vector2d;

    ChessBoard(int _rows = 7, int _cols = 10, double _step = 35.0)
        : rows(_rows)
        , cols(_cols)
        , step(_step)
    {
        options.x_shift = options.y_shift = -step / 2.0;
    }

    uint8_t pixVal(double x, double y)
    {
        x = x * options.scale + options.x_shift;
        y = y * options.scale + options.y_shift;

        int col = cvFloor(x / step) + 1;
        int row = cvFloor(y / step) + 1;
        if (col < 0 || col > cols
            || row < 0 || row > rows)
            return options.back_ground;
        else
            return (((row + col) & 1) ^ options.start_with_white) ? 200 : 50;
    }

    void draw(cv::Mat& image)
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
    };

public:
    const int rows, cols; //!< num of corners
    const double step;

    PixelOptions options;
};

} //namespace bxg
