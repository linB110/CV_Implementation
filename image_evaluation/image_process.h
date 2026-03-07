#include <opencv2/opencv.hpp>
#include <iostream>

class ImageMetric
{
    public:
    // construct input image for class
    void setImage(const cv::Mat& input_img);
    
    // metric evaluation operations
    float illumination();        // S1
    float contrast();            // S2
    float blur();                // S3
    float spatialEntropy();      // S4
    float repetitivePattern();   // S5
    float noise();               // S6

    private:

    // input image
    cv::Mat img;
};
