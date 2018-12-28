//
// Created by mmc-wy on 18-12-28.
//

#include "hsvThrehold.h"

/*
 * 通过将h和v相加可以达到克服阴影的效果
 */
cv::Mat hsvThrehold::hsv(cv::Mat img)
{
    //通过将h和v相加可以达到克服阴影的效果
    cv::Mat HSV;
    cv::cvtColor(img, HSV, CV_BGR2HSV);
    std::vector<cv::Mat> channels;
    cv::split(HSV, channels);

    cv::Mat temp;
    temp = channels.at(2) + channels.at(0);


    cv::threshold(temp, temp, 0, 255, CV_THRESH_TRIANGLE);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));
    cv::erode(temp, temp, kernel);
    cv::dilate(temp, temp, kernel);
    cv::dilate(temp, temp, kernel);
    cv::dilate(temp, temp, kernel);
    return temp;
}