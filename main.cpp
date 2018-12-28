#include <opencv2/opencv.hpp>
//#include <opencv2/legacy/legacy.hpp>
#include <iostream>
#include "src/BinaryMethod/NiblackBinarization.h"
#include "src/BinaryMethod/SauvolaBinarization.h"
#include "src/BinaryMethod/hsvThrehold.h"
#include "src/BinaryMethod/calc.h"


void showImg(std::string name, cv::Mat& src);
void BrightnessAndContrastAuto(const cv::Mat &src, cv::Mat &dst, float clipHistPercent);
void unevenLightCompensate(cv::Mat &image, int blockSize);



cv::Mat logTransform1(cv::Mat srcImage, int c)
{
    if (srcImage.empty())
        std::cout << "No data" << std::endl;
    cv::Mat resultImage = cv::Mat::zeros(srcImage.size(), srcImage.type());
    cv::add(srcImage, cv::Scalar(1.0), srcImage);  //计算 r+1
    srcImage.convertTo(srcImage, CV_32F);  //转化为32位浮点型
    log(srcImage, resultImage);            //计算log(1+r)
    resultImage = c*resultImage;
    //归一化处理
    normalize(resultImage, resultImage, 0, 255, cv::NORM_MINMAX);
    convertScaleAbs(resultImage, resultImage);
    return resultImage;

}
/*
 * 自适应阈值分割函数--自定义,和opencv的相似,但是比opencv多一个中值滤波
 */

enum METHOD {MEAN,GAUSS,MEDIA};
cv::Mat adaptiveThresh(cv::Mat I, int radius, float ratio, METHOD method = MEAN)
{
    //1. 对图像平滑处理
    cv::Mat smooth;
    switch (method)
    {
        case MEAN:
            cv::boxFilter(I,smooth,CV_32FC1,cv::Size(2*radius+1,2*radius+1));
            break;
        case GAUSS:
            cv::GaussianBlur(I,smooth,cv::Size(2*radius+1,2*radius+1),0,0);
            break;
        case MEDIA:
            cv::medianBlur(I,smooth,2*radius+1);
            break;
        default:
            break;
    }
    //2. 平滑结果乘以比例系数,然后图像矩阵与其作差
    I.convertTo(I,CV_32FC1);
    smooth.convertTo(smooth,CV_32FC1);
    cv::Mat diff = I - (1.0 - ratio)*smooth;
    //3. 阈值处理,大于0输出255,小于0输出0;
    cv::Mat out = cv::Mat::zeros(diff.size(),CV_8UC1);
    for (int r = 0; r < out.rows; ++r) {
        for (int c = 0; c < out.cols; ++c) {

            if(diff.at<float>(r,c) >=0)
                out.at<uchar>(r,c) = 255;
        }
    }
    return out;
}


int main()
{
    cv::Mat img = cv::imread("/home/mmc-wy/Videos/chenzhou20181218-mmc/S5640037.JPG",1);//DJI_0006.JPG  S5640037.JPG
                                //    BrightnessAndContrastAuto(img,img,1);
                                //    showImg("BrightnessAndContrastAuto",img);
                                //    cv::cvtColor(img,img,CV_BGR2GRAY);

    showImg("origin",img);
    cv::Mat gray ;
    cv::cvtColor(img,gray,cv::COLOR_BGR2GRAY);
    cv::Mat lab;
    cv::cvtColor(img,lab,cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> LabChannels;
    cv::split(lab, LabChannels);
//    showImg("L_", LabChannels.at(0));
//    showImg("a", LabChannels.at(1));
//    showImg("b", LabChannels.at(2));

    cv::Mat luv;
    cv::cvtColor(img,luv,cv::COLOR_BGR2Luv);
    std::vector<cv::Mat> LuvChannels;
    cv::split(luv, LuvChannels);
//    showImg("L", LuvChannels.at(0));
//    showImg("u", LuvChannels.at(1));
//    showImg("v", LuvChannels.at(2));

    cv::Mat hsv;
    cv::cvtColor(img,hsv,cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> hsvChannels;
    cv::split(hsv, hsvChannels);
//    showImg("h", hsvChannels.at(0));
//    showImg("s", hsvChannels.at(1));
//    showImg("v_", hsvChannels.at(2));

    cv::Mat xyz;
    cv::cvtColor(img,xyz,cv::COLOR_BGR2XYZ);
    std::vector<cv::Mat> xyzChannels;
    cv::split(xyz, xyzChannels);
//    showImg("x", xyzChannels.at(0));
//    showImg("y", xyzChannels.at(1));
//    showImg("z", xyzChannels.at(2));

    cv::Mat temp = hsvChannels.at(0) + xyzChannels.at(2);
    showImg("temp",temp);
    cv::threshold(temp,temp,0,255,cv::THRESH_TRIANGLE);
//    cv::threshold(temp,temp,0,255,cv::THRESH_BINARY_INV);
//    temp = gray - temp;
    showImg("threshold",temp);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(9,9));
//    cv::Mat kernel2 = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(5,5));
    cv::erode(temp,temp,kernel);
    cv::dilate(temp,temp,kernel);
    cv::erode(temp,temp,kernel);
    cv::erode(temp,temp,kernel);
    cv::dilate(temp,temp,kernel);
    cv::dilate(temp,temp,kernel);
    cv::dilate(temp,temp,kernel);
    showImg("erode",temp);
    calc c;
    double angle = c.GetRotateAngle(temp);
    showImg("calc",temp);
    std::cout<<angle<<std::endl;











                                {
                                    //    自适应阈值分割
                                    //    cv::Mat out = adaptiveThresh(gray,150,0,MEDIA);
                                    //    showImg("adaptiveThresh",out);
                                    //    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));
                                    //    cv::erode(out,out,kernel);
                                    //    showImg("erode",out);
                                }

    
                                //    cv::Mat gray;
                                //    cv::cvtColor(img,gray,cv::COLOR_BGR2GRAY);//灰度变换
                                //    cv::GaussianBlur(gray,gray,cv::Size(5,5),2);   //高斯滤波
                                //    cv::Canny(gray,gray,40,100);
                                //    showImg("GaussianBlur",gray);





                                    {
                                        //均值漂移 opencv 速度较慢 第3\4个参数越大,模糊效果越好
                                        //    cv::Mat dst;
                                        //    cv::pyrMeanShiftFiltering(img,dst,50,30);
                                        //    showImg("pyrMeanShiftFiltering",dst);
                                    }



                                    {
                                        //模糊操作
                                        //    cv::Mat dst;
                                        //    cv::medianBlur(img,dst,11);
                                        //    showImg("medianBlur",dst);
                                    }


//    cv::Mat gray;
//    cv::cvtColor(img,gray,CV_BGR2GRAY);
//    showImg("gray",gray);
//    cv::Mat res = logTransform1(gray,3);
//    showImg("logTransform1",res);



//    std::vector<cv::Mat> BGR;
//    cv::split(img,BGR);
//    showImg("B",BGR.at(0));
//    showImg("G",BGR.at(1));
//    showImg("R",BGR.at(2));
                                    {
                                        //通过将h和v相加可以达到克服阴影的效果
//                                        cv::Mat HSV, Lab;
//                                        cv::cvtColor(img, HSV, CV_BGR2HSV);
//                                        std::vector<cv::Mat> channels;
//                                        cv::split(HSV, channels);
//                                        showImg("h", channels.at(0));
//                                        showImg("s", channels.at(1));
//                                        showImg("v", channels.at(2));
//                                        cv::Mat temp;
//                                        temp = channels.at(2) + channels.at(0);
//                                        showImg("temp", temp);
//
//                                        cv::threshold(temp, temp, 0, 255, CV_THRESH_TRIANGLE);
//                                        showImg("temp2", temp);
//                                        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 11));
//                                        cv::Mat kernel2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));
//                                        cv::erode(temp, temp, kernel);
//                                        cv::dilate(temp, temp, kernel2);
//                                        cv::dilate(temp, temp, kernel2);
//                                        cv::dilate(temp, temp, kernel2);
//                                        showImg("temp3", temp);

//                                            hsvThrehold hsvBinary;
//                                            cv::Mat out = hsvBinary.hsv(img);
//                                            showImg("hsv",out);

                                    }



//    showImg("gray",gray);
//    cv::Mat thre;
//    cv::threshold(gray,thre,115,255,CV_THRESH_BINARY_INV);
//    showImg("threhold",thre);
//    cv::Mat temp = gray - thre;

//    showImg("temp",temp);

                                    {
                                        //Niblack Sauvola二值化方法
                                //    auto nb = new ImageBinarization::NiblackBinarization();
                                //    auto sb = new ImageBinarization::SauvolaBinarization();
                                //    cv::Mat dst;
                                //    nb->Binarize(img,dst,200);
                                //    sb->Binarize(img,dst,3000,-0.2);
                                //    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(11,11));
                                //    cv::Mat kernel2 = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(13,13));
                                //    cv::morphologyEx(dst,dst,cv::MORPH_OPEN,kernel);
                                //    cv::morphologyEx(dst,dst,cv::MORPH_CLOSE,kernel2);

                                //    cv::erode(dst,dst,kernel);
                                //    cv::erode(dst,dst,kernel);
                                //
                                //    cv::dilate(dst,dst,kernel);
                                        //cv::dilate(dst,dst,kernel);
                                //    showImg("dst",dst);
                                    }

                                    {
                                        //处理光照不均匀情形
                                //    unevenLightCompensate(img,160);
                                //    showImg("unevenLightCompensate",img);
                                //    cv::threshold(img,img,0,255,cv::THRESH_OTSU);
                                //    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));
                                //    cv::erode(img,img,kernel);
                                //    cv::dilate(img,img,kernel);
                                //    cv::dilate(img,img,kernel);
                                //    showImg("img",img);
                                    }

    /*
     *
     *  image
     *      单通道输入图像.
     *  edges
     *      单通道存储边缘的输出图像
     *  threshold1
     *      第一个阈值
     *  threshold2
     *      第二个阈值
     *  aperture_size
     *      Sobel 算子内核大小
     */
//    cv::Canny( gray, gray, 50, 80, 3 );//边缘检测
//    showImg("temp",gray);
    cv::waitKey(0);
    return 0;
}





void unevenLightCompensate(cv::Mat &image, int blockSize)
{
    if (image.channels() == 3) cvtColor(image, image, 7);
    double average = mean(image)[0];
    int rows_new = ceil(double(image.rows) / double(blockSize));
    int cols_new = ceil(double(image.cols) / double(blockSize));
    cv::Mat blockImage;
    blockImage = cv::Mat::zeros(rows_new, cols_new, CV_32FC1);
    for (int i = 0; i < rows_new; i++)
    {
        for (int j = 0; j < cols_new; j++)
        {
            int rowmin = i*blockSize;
            int rowmax = (i + 1)*blockSize;
            if (rowmax > image.rows) rowmax = image.rows;
            int colmin = j*blockSize;
            int colmax = (j + 1)*blockSize;
            if (colmax > image.cols) colmax = image.cols;
            cv::Mat imageROI = image(cv::Range(rowmin, rowmax), cv::Range(colmin, colmax));
            double temaver = mean(imageROI)[0];
            blockImage.at<float>(i, j) = temaver;
        }
    }
    blockImage = blockImage - average;
    cv::Mat blockImage2;
    resize(blockImage, blockImage2, image.size(), (0, 0), (0, 0), cv::INTER_CUBIC);
    cv::Mat image2;
    image.convertTo(image2, CV_32FC1);
    cv::Mat dst = image2 - blockImage2;
    dst.convertTo(image, CV_8UC1);
}



void showImg(std::string name, cv::Mat& src)
{
    cv::namedWindow(name,CV_WINDOW_NORMAL);
    cv::resizeWindow(name,800,600);
    cv::imshow(name,src);
}

void BrightnessAndContrastAuto(const cv::Mat &src, cv::Mat &dst, float clipHistPercent)
{
    CV_Assert(clipHistPercent >= 0);
    CV_Assert((src.type() == CV_8UC1) || (src.type() == CV_8UC3) || (src.type() == CV_8UC4));

    int histSize = 256;
    float alpha, beta;
    double minGray = 0, maxGray = 0;

    //to calculate grayscale histogram
    cv::Mat gray;
    if (src.type() == CV_8UC1) gray = src;
    else if (src.type() == CV_8UC3) cv::cvtColor(src, gray, CV_BGR2GRAY);
    else if (src.type() == CV_8UC4) cv::cvtColor(src, gray, CV_BGRA2GRAY);
    if (clipHistPercent == 0)
    {
        // keep full available range
        cv::minMaxLoc(gray, &minGray, &maxGray);
    }
    else
    {
        cv::Mat hist; //the grayscale histogram

        float range[] = { 0, 256 };
        const float* histRange = { range };
        bool uniform = true;
        bool accumulate = false;
        cv::calcHist(&gray, 1, 0, cv::Mat (), hist, 1, &histSize, &histRange, uniform, accumulate);

        // calculate cumulative distribution from the histogram
        std::vector<float> accumulator(histSize);
        accumulator[0] = hist.at<float>(0);
        for (int i = 1; i < histSize; i++)
        {
            accumulator[i] = accumulator[i - 1] + hist.at<float>(i);
        }

        // locate points that cuts at required value
        float max = accumulator.back();
        clipHistPercent *= (max / 100.0); //make percent as absolute
        clipHistPercent /= 2.0; // left and right wings
        // locate left cut
        minGray = 0;
        while (accumulator[minGray] < clipHistPercent)
            minGray++;

        // locate right cut
        maxGray = histSize - 1;
        while (accumulator[maxGray] >= (max - clipHistPercent))
            maxGray--;
    }

    // current range
    float inputRange = maxGray - minGray;

    alpha = (histSize - 1) / inputRange;   // alpha expands current range to histsize range
    beta = -0.5*minGray * alpha;             // beta shifts current range so that minGray will go to 0

    // Apply brightness and contrast normalization
    // convertTo operates with saurate_cast
    src.convertTo(dst, -1, alpha, beta);

    // restore alpha channel from source
    if (dst.type() == CV_8UC4)
    {
        int from_to[] = { 3, 3};
        cv::mixChannels(&src, 4, &dst,1, from_to, 1);
    }
    return;
}