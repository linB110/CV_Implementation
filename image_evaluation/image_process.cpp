#include <math.h>
#include <numeric>
#include <opencv2/opencv.hpp>
#include "image_process.h"

#include <iostream>


void ImageMetric::setImage(const cv::Mat& input_img)
{
    if(input_img.channels() == 3)
    {
        cv::cvtColor(input_img, img, cv::COLOR_BGR2GRAY);
    }
        
    else
    {
        img = input_img.clone();
    }
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
    cv::Laplacian(img, lap, CV_64F, 3);

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
    cv::magnitude(sobel_x, sobel_y, magnitude);   // sqrt(gx^2 + gy^2)

    // histogram parameters
    int histSize = 256;
    float range[] = {0,256};
    const float* histRange = {range};

    cv::Mat hist;
    cv::calcHist(&magnitude,1,0,cv::Mat(),hist,1,&histSize,&histRange);

    // convert histogram to probability distribution
    hist /= magnitude.total();

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
    const int G = 64;

    cv::Mat quantized;
    img.convertTo(quantized, CV_32F);

    quantized = quantized * G / 256.0f;
    quantized.setTo(G-1, quantized >= G); 

    std::vector<std::pair<int,int>> directions =
    {
        {1,0},
        {0,1},
        {1,1},
        {-1,1}
    };

    float total_energy = 0.0f;

    for(auto d : directions)
    {
        int dx = d.first;
        int dy = d.second;

        cv::Mat glcm = cv::Mat::zeros(G,G,CV_32F);

        for(int y=0;y<quantized.rows;y++)
        {
            for(int x=0;x<quantized.cols;x++)
            {
                int nx = x + dx;
                int ny = y + dy;

                if(nx < 0 || nx >= quantized.cols ||
                   ny < 0 || ny >= quantized.rows)
                    continue;

                int i = quantized.at<float>(y,x);
                int j = quantized.at<float>(ny,nx);

                i = std::min(i,G-1);
                j = std::min(j,G-1);

                glcm.at<float>(i,j) += 1.0f;
                glcm.at<float>(j,i) += 1.0f;   // symmetric
            }
        }

        double sum = cv::sum(glcm)[0];

        if(sum > 0)
            glcm /= sum;

        float energy = 0.0f;

        for(int i=0;i<G;i++)
        {
            for(int j=0;j<G;j++)
            {
                float p = glcm.at<float>(i,j);
                energy += p*p;
            }
        }

        total_energy += energy;
    }

    return total_energy / directions.size();
}

// S6 score
float ImageMetric::noise()
{
    cv::Mat img_copy;

    // normalize 0~1
    img.convertTo(img_copy, CV_32F, 1.0/255.0); 

    cv::Mat median;
    cv::medianBlur(img_copy, median, 3);
    cv::Mat diff = cv::abs(img_copy - median);
    
    return (float)cv::mean(diff)[0];
}

float ImageMetric::computePCC(const std::vector<float>& x,
                              const std::vector<float>& y)
{
    int n = x.size();
    if (n != y.size() || n == 0) return 0;

    float mean_x = 0, mean_y = 0;
    for (int i = 0; i < n; ++i) {
        mean_x += x[i];
        mean_y += y[i];
    }
    mean_x /= n;
    mean_y /= n;

    float num = 0, den_x = 0, den_y = 0;
    for (int i = 0; i < n; ++i) {
        float dx = x[i] - mean_x;
        float dy = y[i] - mean_y;
        num   += dx * dy;
        den_x += dx * dx;
        den_y += dy * dy;
    }

    float den = std::sqrt(den_x * den_y);
    if (den < 1e-8f) return 0;
    return num / den;
}

std::vector<float> ImageMetric::normalizeVector(const std::vector<float>& v)
{
    float min_v = *std::min_element(v.begin(), v.end());
    float max_v = *std::max_element(v.begin(), v.end());

    std::vector<float> res(v.size());
    for(int i=0;i<v.size();i++)
        res[i] = (v[i] - min_v) / (max_v - min_v + 1e-8);

    return res;
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

    // assign P -> PCC matrix for use
    PCCMatrix = P;

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

std::vector<std::vector<float>> ImageMetric::normalizeMetrics(
    const std::vector<std::vector<float>>& scores)
{
    int K = scores.size();
    int N = scores[0].size();

    std::vector<std::vector<float>> Sn(K, std::vector<float>(N));

    for(int j=0;j<K;j++)
    {
        float min_v = *std::min_element(scores[j].begin(), scores[j].end());
        float max_v = *std::max_element(scores[j].begin(), scores[j].end());

        for(int i=0;i<N;i++)
        {
            Sn[j][i] = (scores[j][i] - min_v) / (max_v - min_v + 1e-8);
        }
    }

    return Sn;
}

float ImageMetric::stddev(const std::vector<float>& v)
{
    int n = v.size();

    float mean = 0;
    for(float x : v)
        mean += x;

    mean /= n;

    float var = 0;
    for(float x : v)
        var += (x-mean)*(x-mean);

    var /= (n-1);

    return sqrt(var);
}

std::vector<float> ImageMetric::computeCRITICWeights(
        const std::vector<std::vector<float>>& Sn)
{
    int M = Sn.size();
    int N = Sn[0].size();

    std::vector<float> sigma(M,0);
    std::vector<float> C(M,0);

    // compute std
    for(int j=0;j<M;j++)
    {
        float mean = 0;

        for(int i=0;i<N;i++)
            mean += Sn[j][i];

        mean /= N;

        float var = 0;

        for(int i=0;i<N;i++)
        {
            float d = Sn[j][i] - mean;
            var += d*d;
        }

        sigma[j] = sqrt(var/N);
    }

    // PCC matrix
    std::vector<std::vector<float>> corr(M,std::vector<float>(M));

    for(int j=0;j<M;j++)
        for(int l=0;l<M;l++)
            corr[j][l] = computePCC(Sn[j],Sn[l]);

    // compute Cj
    for(int j=0;j<M;j++)
    {
        float sum = 0;

        for(int l=0;l<M;l++)
            sum += (1 - corr[j][l]);

        C[j] = sigma[j] * sum;
    }

    float total = std::accumulate(C.begin(),C.end(),0.0f);

    std::vector<float> w(M);

    for(int j=0;j<M;j++)
        w[j] = C[j] / total;

    return w;
}

std::vector<float> ImageMetric::computeBlockScore(
        const std::vector<std::vector<float>>& Sn,
        const std::vector<float>& w)
{
    int M = Sn.size();
    int N = Sn[0].size();

    std::vector<float> score(N,0);

    for(int i=0;i<N;i++)
    {
        for(int j=0;j<M;j++)
            score[i] += w[j] * Sn[j][i];
    }

    return score;
}
