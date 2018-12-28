//
// Created by mmc-wy on 18-12-28.
//

#ifndef SAUVOLA_CALC_H
#define SAUVOLA_CALC_H

#include <iostream>
#include <opencv2/opencv.hpp>

class calc {
public:
    double getMinEnclosingTriangle(std::vector<cv::Point> &pts, cv::Mat& img, int imgWidth, int imgHeight);
    double GetRotateAngle(cv::Mat& frame);

};


#endif //SAUVOLA_CALC_H
