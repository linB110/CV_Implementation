#include <DBoW2/DBoW2.h>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <algorithm>

using namespace std;
using namespace cv;

typedef DBoW2::BowVector BowVector;

vector<string> getImageFiles(const string& folder) 
{
    vector<string> files;
    DIR* dir = opendir(folder.c_str());
    struct dirent* ent;

    if (dir != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            string fname = ent->d_name;
            if (fname.find(".jpg") != string::npos ||
                fname.find(".png") != string::npos ||
                fname.find(".ppm") != string::npos) {
                files.push_back(folder + "/" + fname);
            }
        }
        closedir(dir);
    }

    sort(files.begin(), files.end());  
    return files;
}

vector<BowVector> extractBOW(const vector<string>& img_paths, OrbVocabulary& voc) 
{
    vector<BowVector> bow_vecs;
    Ptr<Feature2D> orb = ORB::create(1000);

    for (const auto& path : img_paths) {
        Mat img = imread(path, IMREAD_GRAYSCALE);
        if (img.empty()) continue;

        vector<KeyPoint> keypoints;
        Mat descriptors;
        orb->detectAndCompute(img, noArray(), keypoints, descriptors);
        if (descriptors.empty()) continue;

        vector<Mat> desc_vec;
        for (int i = 0; i < descriptors.rows; ++i) {
            desc_vec.push_back(descriptors.row(i));
        }

        BowVector bow;
        DBoW2::FeatureVector feat_vec;
        voc.transform(desc_vec, bow);
        bow_vecs.push_back(bow);
    }

    return bow_vecs;
}

void evaluateVocabulary(const string& voc_file, const vector<string>& img_paths) 
{
    cout << "📂 Evaluating with: " << voc_file << endl;

    OrbVocabulary voc;
    voc.load(voc_file);

    vector<BowVector> bows = extractBOW(img_paths, voc);
    int N = bows.size();

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double score = voc.score(bows[i], bows[j]);
            cout << fixed << setprecision(4) << score << " ";
        }
        cout << endl;
    }

    cout << "----------------------------------------" << endl;
}

int main() 
{
    cout << "evaluate illumination -- fog " << endl; 
    
    string img_folder1 = "/home/lab605/dataset/hpatches-sequences-release/v_london";
    vector<string> img_paths = getImageFiles(img_folder1);

    evaluateVocabulary("voc_v_london.yml.gz", img_paths);
    evaluateVocabulary("voc_i_bridger.yml.gz", img_paths);
    evaluateVocabulary("voc_all.yml.gz", img_paths);
    
    cout << "evaluate viewpoint -- london " << endl; 
        
    string img_folder2 = "/home/lab605/dataset/hpatches-sequences-release/v_london";
    img_paths = getImageFiles(img_folder2);

    evaluateVocabulary("voc_v_london.yml.gz", img_paths);
    evaluateVocabulary("voc_i_bridger.yml.gz", img_paths);
    evaluateVocabulary("voc_all.yml.gz", img_paths);

    return 0;
}

