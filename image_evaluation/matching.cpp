#include <math.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "image_process.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        cout << "need image path\n";
        return -1;
    }

    string image_path = argv[1];

    Mat img = imread(image_path);

    if(img.empty())
    {
        cout << "image load failed\n";
        return -1;
    }

    Mat gray;
    if(img.channels() == 3)
    {
        cvtColor(img, gray, COLOR_BGR2GRAY);
    }else
    {
        gray = img.clone();
    }

    int block_size = 8;

    vector<float> S1_list;
    vector<float> S2_list;
    vector<float> S3_list;
    vector<float> S4_list;
    vector<float> S5_list;
    vector<float> S6_list;

    vector<Rect> blocks;

    ImageMetric metric;

    // -------------------------
    // 1. compute metrics
    // -------------------------

    for(int y=0; y+block_size<=gray.rows; y+=block_size)
    {
        for(int x=0; x+block_size<=gray.cols; x+=block_size)
        {
            Rect roi(x,y,block_size,block_size);

            Mat block = gray(roi);

            metric.setImage(block);

            S1_list.push_back(metric.illumination());
            S2_list.push_back(metric.contrast());
            S3_list.push_back(metric.blur());
            S4_list.push_back(metric.spatialEntropy());
            S5_list.push_back(metric.repetitivePattern());
            S6_list.push_back(metric.noise());

            blocks.push_back(roi);
        }
    }

    cout<<"Total blocks: "<<S1_list.size()<<endl;

    vector<vector<float>> metrics =
    {
        S1_list,
        S2_list,
        S3_list,
        S4_list,
        S5_list,
        S6_list
    };

    // -------------------------
    // 2. metric selection
    // -------------------------

    float threshold = 0.5;

    vector<int> selected =
        metric.metricSelection(metrics,threshold);

    cout<<"\nSelected metrics:\n";

    for(int i:selected)
        cout<<"S"<<i+1<<" ";

    cout<<endl;

    // PCC Matrix
    const auto& P = metric.PCCMatrix;
    cout << "\nPCC Matrix :\n"; 
    for(int i=0;i<P.size();i++)
    {
        for(int j=0;j<P[i].size();j++)
        {
            printf("%8.4f ", P[i][j]);
        }
        cout << endl;
    }

    // build selected metric matrix

    vector<vector<float>> selected_metrics;

    for(int idx:selected)
        selected_metrics.push_back(metrics[idx]);

    // -------------------------
    // 3. CRITIC weights
    // -------------------------

    vector<float> weights =
        metric.computeCRITICWeights(selected_metrics);

    cout<<"\nCRITIC Weights\n";

    for(int i=0;i<weights.size();i++)
        cout<<"w"<<i+1<<" = "<<weights[i]<<endl;

    // -------------------------
    // 4. block scores
    // -------------------------

    vector<float> scores =
        metric.computeBlockScore(
            selected_metrics,
            weights
        );

    // -------------------------
    // 5. visualization
    // -------------------------

    Mat vis = img.clone();

    for(int i=0;i<scores.size();i++)
    {
        float s = scores[i];

        Scalar color;

        if(s < 0.3)
            color = Scalar(0,0,255);      // red
        else if(s <= 0.5)
            color = Scalar(0,255,255);    // yellow
        else
            color = Scalar(0,255,0);      // green

        rectangle(vis,blocks[i],color,2);

        string text = format("%.2f",s);

        putText(
            vis,
            text,
            Point(blocks[i].x+4,
                  blocks[i].y+18),
            FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1
        );
    }

    imshow("Block Weight Visualization",vis);
    
    while(true)
    { 
        int key = cv::waitKey(0); 
        if(key == 'q' || key == 'Q')   
            break;
    }    
    cv::destroyAllWindows();

    return 0;
}
