#include <math.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "image_process.h"

void ImageMetric::setImage(const cv::Mat& input_img)
{
    img = input_img;
}

// S1 score
float ImageMetric::illumination()
{
    cv::Scalar mean = cv::mean(img);
    float I_mean = mean[0];

    return std::abs(I_mean - 128.0f);
}

// S2 score
float ImageMetric::contrast()
{
    cv::Scalar mean, stddev;
    cv::meanStdDev(img, mean, stddev);

    return stddev[0];
}

// S3 score
float ImageMetric::blur()
{
    cv::Mat lap;
    cv::Laplacian(img, lap, CV_64F);

    cv::Scalar mean, stddev;
    cv::meanStdDev(lap, mean, stddev);

    return stddev[0] * stddev[0];
}

// S4 score
float ImageMetric::spatialEntropy()
{
    cv::Mat sobel_x, sobel_y;
    
    // compute Sobel gradients
    cv::Sobel(img, sobel_x, CV_32F, 1, 0, 3);
    cv::Sobel(img, sobel_y, CV_32F, 0, 1, 3);

    // gradient magnitude
    cv::Mat magnitude;
    cv::magnitude(sobel_x, sobel_y, magnitude);

    // normalize to 0~255
    cv::Mat mag8;
    magnitude.convertTo(mag8, CV_8U);

    // histogram parameters
    int histSize = 256;
    float range[] = {0,256};
    const float* histRange = {range};

    cv::Mat hist;
    cv::calcHist(&mag8,1,0,cv::Mat(),hist,1,&histSize,&histRange);

    // convert histogram to probability distribution
    hist /= mag8.total();

    float entropy = 0.0f;

    for(int i=0;i<histSize;i++)
    {
        float p = hist.at<float>(i);

        if(p > 1e-12)
            entropy -= p * log2(p);
    }

    return entropy;
}

// S5 score
float ImageMetric::repetitivePattern()
{
    const int G = 256;
    
    // gray-level co-occurence matrix
    cv::Mat glcm = cv::Mat::zeros(G,G,CV_32F);
    
    // C(i,j)
    for(int y=0;y<img.rows;y++)
    {
        for(int x=0;x<img.cols-1;x++)
        {
            int i = img.at<uchar>(y,x);
            int j = img.at<uchar>(y,x+1);

            glcm.at<float>(i,j)++;
        }
    }

    glcm /= cv::sum(glcm)[0];

    float energy = 0;

    // P(i,j)
    for(int i=0;i<G;i++)
        for(int j=0;j<G;j++)
        {
            float p = glcm.at<float>(i,j);
            energy += p*p;
        }

    return energy;
}

// S6 score
float ImageMetric::noise()
{
    cv::Mat median;
    cv::medianBlur(img, median, 3);

    cv::Mat diff;
    cv::absdiff(img, median, diff);

    return cv::mean(diff)[0];
}