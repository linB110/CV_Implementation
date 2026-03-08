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
    // quantized gray levels
    const int G = 64;   

    cv::Mat quantized;
    img.convertTo(quantized, CV_8U);

    // quantize image
    quantized = quantized / (256 / G);

    cv::Mat glcm = cv::Mat::zeros(G, G, CV_32F);

    int dx = 3;
    int dy = 3;

    for(int y = 0; y < quantized.rows - dy; y++)
    {
        for(int x = 0; x < quantized.cols - dx; x++)
        {
            int i = quantized.at<uchar>(y,x);
            int j = quantized.at<uchar>(y+dy,x+dx);

            i = std::min(i, G-1);
            j = std::min(j, G-1);

            glcm.at<float>(i,j) += 1.0f;
        }
    }

    // normalize to probability
    double s = cv::sum(glcm)[0];
    if(s > 0)
        glcm /= s;

    float energy = 0.0f;

    for(int i=0;i<G;i++)
    {
        for(int j=0;j<G;j++)
        {
            float p = glcm.at<float>(i,j);
            energy += p * p;
        }
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

float ImageMetric::computePCC(const std::vector<float>& x, const std::vector<float>& y)
{
    int n = x.size();
    if(n != y.size() || n == 0) 
        return 0;

    float mean_x = 0, mean_y = 0;
    for(int i=0;i<n;i++)
    {
        mean_x += x[i];
        mean_y += y[i];
    }

    mean_x /= n;
    mean_y /= n;

    float num = 0;
    float den_x = 0;
    float den_y = 0;

    for(int i=0;i<n;i++)
    {
        float dx = x[i] - mean_x;
        float dy = y[i] - mean_y;

        num += dx * dy;
        den_x += dx * dx;
        den_y += dy * dy;
    }

    return fabs(num / sqrt(den_x * den_y + 1e-8));
}

float ImageMetric::computeVariance(const std::vector<float>& x)
{
    int n = x.size();
    if(n == 0) 
        return 0;

    float mean = 0;
    for(float v : x)
        mean += v;

    mean /= n;

    float var = 0;
    for(float v : x)
    {
        float d = v - mean;
        var += d * d;
    }

    return var / (n - 1 + 1e-8);
}

float ImageMetric::computeConditionalCovariance(const std::vector<float>& x, const std::vector<float>& y)
{
    int n = x.size();
    if(n != y.size() || n == 0) return 0;

    float mean_x = 0, mean_y = 0;

    for(int i=0;i<n;i++)
    {
        mean_x += x[i];
        mean_y += y[i];
    }

    mean_x /= n;
    mean_y /= n;

    float var_x = 0;
    float var_y = 0;
    float cov_xy = 0;

    for(int i=0;i<n;i++)
    {
        float dx = x[i] - mean_x;
        float dy = y[i] - mean_y;

        var_x += dx * dx;
        var_y += dy * dy;
        cov_xy += dx * dy;
    }

    var_x /= (n-1);
    var_y /= (n-1);
    cov_xy /= (n-1);

    float cond = var_x - (cov_xy / (var_y + 1e-8) * cov_xy) ;

    return cond;
}

float ImageMetric::computeMutualInformation(const std::vector<float>& x, const std::vector<float>& y)
{
    float var_x = computeVariance(x);
    float cond = computeConditionalCovariance(x,y);

    if(cond <= 0) 
        cond = 1e-8;

    return 0.5f * log2(var_x / cond);
}

std::vector<int> ImageMetric::metricSelection(std::vector<std::vector<float>>& metrics, float Threshold)
{
    int M = metrics.size();   // should be 6

    // -------------------
    // Step1: PCC matrix
    // -------------------

    std::vector<std::vector<float>> P(M, std::vector<float>(M,0));

    for(int i=0;i<M;i++)
    {
        for(int j=0;j<M;j++)
        {
            P[i][j] = computePCC(metrics[i], metrics[j]);
        }
    }

    // -------------------
    // Step2: determine K
    // -------------------

    int K = M;
    std::set<int> removed;
    for(int i=1;i<M;i++)
    {
        for(int j=0;j<i;j++)
        {
            if(P[i][j] > Threshold)
                removed.insert(j);
        }
    }

    K = M - removed.size();
    if(K < 1) 
        K = 1;

    // -------------------
    // Step3: first metric
    // max variance
    // -------------------

    std::vector<int> selected;
    std::vector<int> remaining;

    for(int i=0;i<M;i++)
        remaining.push_back(i);

    int best = -1;
    float best_var = -1;

    for(int i=0;i<M;i++)
    {
        float v = computeVariance(metrics[i]);

        if(v > best_var)
        {
            best_var = v;
            best = i;
        }
    }

    selected.push_back(best);

    remaining.erase(
        std::remove(remaining.begin(), remaining.end(), best),
        remaining.end()
    );

    // -------------------
    // Step4: iterative MI
    // -------------------

    while(selected.size() < K)
    {
        int best_i = -1;
        float best_mi = -1;

        for(int idx : remaining)
        {
            float mi_sum = 0;

            for(int s : selected)
            {
                mi_sum += computeMutualInformation(
                    metrics[idx],
                    metrics[s]
                );
            }

            if(mi_sum > best_mi)
            {
                best_mi = mi_sum;
                best_i = idx;
            }
        }

        selected.push_back(best_i);

        remaining.erase(
            std::remove(remaining.begin(), remaining.end(), best_i),
            remaining.end()
        );
    }

    return selected;
}
