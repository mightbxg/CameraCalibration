#pragma once

#include <opencv2/opencv.hpp>

namespace bxg {

class ChessBoard {
public:
    ChessBoard(int _rows = 7, int _cols = 10, double _step = 35.0)
        : rows(_rows)
        , cols(_cols)
        , step(_step)
    {
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
            return (((row + col) & 1) ^ options.start_with_white) * 255;
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

public:
    struct PixelOptions {
        double x_shift { 0.0 }, y_shift { 0.0 };
        double scale { 1.0 };
        bool start_with_white { true };
        uint8_t back_ground { 126 };
    };

public:
    const int rows, cols;
    const double step;

    PixelOptions options;
};

} //namespace bxg
