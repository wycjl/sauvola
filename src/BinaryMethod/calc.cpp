//
// Created by mmc-wy on 18-12-28.
//

#include "calc.h"
#include <opencv2/opencv.hpp>

double calc::GetRotateAngle(cv::Mat& frame)
{

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i>hierarchy;
    cv::findContours(frame,contours,hierarchy,CV_RETR_LIST,CV_CHAIN_APPROX_NONE);

    double rotateAngle = 0;
    std::vector<double> area;
    if(contours.size()>0)
    {
        for (size_t i=0; i < contours.size(); ++i) {
            area.push_back(cv::contourArea(contours[i]));
        }
        auto biggest_area = std::max_element(std::begin(area),std::end(area));
        auto index = std::distance(std::begin(area),biggest_area);

        rotateAngle = getMinEnclosingTriangle(contours[index], frame, frame.cols, frame.rows);


        return rotateAngle;
    }
    else
    {
        return -999;
    }

}

double calc::getMinEnclosingTriangle(std::vector<cv::Point> &pts, cv::Mat& img, int imgWidth, int imgHeight)
{
    cv::Mat triangle;

    double rect = cv::minEnclosingTriangle(pts, triangle);

    std::vector<cv::Point2f> p;
    p = cv::Mat_<cv::Point2f>(triangle);
//    triangle.points(P);
    for (int j = 0; j < 3; j++)
    {
        cv::line(img, p[j], p[(j + 1) % 3], cv::Scalar(255,0,0), 5);
    }
    double len[3];
    for (int i = 0; i < 3; i++) {
        len[i] = sqrt(pow(p[i].x - p[(i + 1) % 3].x, 2) + pow(p[i].y - p[(i + 1) % 3].y, 2));
    }
    int index = 0;
    if (len[0] > len[1]) {
        if (len[0] > len[2]) {
            index = 0;
        }
        else {
            index = 2;
        }
    }
    else {
        if (len[1] > len[2]) {
            index = 1;
        }
        else {
            index = 2;
        }
    }
    //index = len[0] > len[1] ? len[0] > len[2] ? 0 : 2 : 1;
    double result = ((p[(index + 2) % 3].x - p[(index + 1) % 3].x)*(p[index].y - p[(index + 1) % 3].y) / (p[index].x - p[(index + 1) % 3].x) + p[(index + 1) % 3].y) - p[(index + 2) % 3].y;
    double angle = ((p[(index) % 3].y - p[(index + 1) % 3].y) / (p[(index) % 3].x - p[(index + 1) % 3].x));
    angle = atan(angle) * 180 / CV_PI;

    if (result < 0 && (p[(index) % 3].y - p[(index + 1) % 3].y) * (p[(index) % 3].x - p[(index + 1) % 3].x) > 0 ){
        //180 -
        angle = angle-180;
    }
    else if(result < 0 && (p[(index) % 3].y - p[(index + 1) % 3].y) * (p[(index) % 3].x - p[(index + 1) % 3].x) < 0){
        angle = 180 + angle;
    }
//    double x0, y0, alpha;

//    x0 = rect.center.x;
//    y0 = rect.center.y;
//    alpha = -rect.angle*CV_PI / 180;
//    int w = 10;
    //
//    line(img, Point2f(x0 - w, y0), Point2f(x0 + w, y0), CV_RGB(0, 255, 0), 2);
//    line(img, Point2f(x0, y0 - w), Point2f(x0, y0 + w), CV_RGB(0, 255, 0), 2);
    //
//    cv::line(img, cv::Point2f(imgWidth / 2 - w, imgHeight / 2), cv::Point2f(imgWidth / 2 + w, imgHeight / 2), CV_RGB(255, 0, 255), 2);
//    cv::line(img, cv::Point2f(imgWidth / 2, imgHeight / 2 - w), cv::Point2f(imgWidth / 2, imgHeight / 2 + w), CV_RGB(255, 0, 255), 2);
//    double angle_3 = -rect.angle;
    ////putText
//    char str[256];
//    sprintf(str, "%f", angle);
//    std::string angle2str = str;
//
//    int font_face = cv::FONT_HERSHEY_COMPLEX;
//    double font_scale = 2;
//    int thickness = 2;
//    //int baseline;
//
//    cv::putText(img, angle2str, (cv::Point2f(imgWidth / 2 - w, imgHeight / 2),cv::Point2f(imgWidth / 2 + w, imgHeight / 2)), font_face, font_scale, cv::Scalar(0, 0, 255), thickness, 8, 0);


    return angle;
}

