#include <math.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "image_process.h"

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cout << "need path to image as input" << std::endl;
    }

    std::string image_path = argv[1];
    cv::Mat img = cv::imread(image_path);

    if(img.empty())
    {
        std::cout << "Failed to load image" << std::endl;
        return -1;
    }
    
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    ImageMetric metric;
    metric.setImage(gray);

     // compute metrics
    float S1 = metric.illumination();
    float S2 = metric.contrast();
    float S3 = metric.blur();
    float S4 = metric.spatialEntropy();
    float S5 = metric.repetitivePattern();
    float S6 = metric.noise();

    // print result
    std::cout << "S1 Illumination: " << S1 << std::endl;
    std::cout << "S2 Contrast: " << S2 << std::endl;
    std::cout << "S3 Blur: " << S3 << std::endl;
    std::cout << "S4 Spatial Entropy: " << S4 << std::endl;
    std::cout << "S5 Repetitive Pattern: " << S5 << std::endl;
    std::cout << "S6 Noise: " << S6 << std::endl;

    cv::imshow("Image", img);
    cv::waitKey(0);

    return 0;
}
