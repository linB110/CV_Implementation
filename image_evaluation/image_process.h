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

    // metric correlation computation
    float computePCC(const std::vector<float>& x, const std::vector<float>& y);

    // feature metric selection
    std::vector<int> metricSelection(std::vector<std::vector<float>>& metrics, float Threshold);

    // compute image block optimizattion weights
    std::vector<float> computeCRITICWeights(const std::vector<std::vector<float>>& scores);
    std::vector<float> computeBlockScore(const std::vector<std::vector<float>>& scores, const std::vector<float>& weights);

    // input image
    cv::Mat img;    
    
    // PCC matrix
    std::vector<std::vector<float>> PCCMatrix;

    private:    
    
    // helper functions for metric calculation and feature selection
    std::vector<float> normalizeVector(const std::vector<float>& v);
    float computeVariance(const std::vector<float>& x);
    float computeConditionalCovariance(const std::vector<float>& x, const std::vector<float>& y);
    float computeMutualInformation(const std::vector<float>& x, const std::vector<float>& y);

    // helper functions for image feature optimization weight
    std::vector<std::vector<float>> normalizeMetrics(const std::vector<std::vector<float>>& scores);
    float stddev(const std::vector<float>& v);
};
