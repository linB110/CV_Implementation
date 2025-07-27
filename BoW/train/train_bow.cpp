#include <DBoW2/DBoW2.h>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <vector>
#include <string>
#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <algorithm>
#include <chrono>

using namespace std;
using namespace cv;
using namespace std::chrono; 


bool isDirectory(const string& path) 
{
    struct stat s;
    return (stat(path.c_str(), &s) == 0) && S_ISDIR(s.st_mode);
}


bool isImageFile(const string& filename, const vector<string>& allowed_exts) 
{
    size_t dot_pos = filename.find_last_of('.');
    if (dot_pos == string::npos) return false;

    string ext = filename.substr(dot_pos);
    transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return find(allowed_exts.begin(), allowed_exts.end(), ext) != allowed_exts.end();
}


vector<vector<Mat>> loadFeatures(const string& root_folder, const vector<string>& allowed_exts) 
{
    vector<vector<Mat>> all_features;
    Ptr<Feature2D> orb = ORB::create(1000);
    DIR* dir = opendir(root_folder.c_str());
    struct dirent* ent;

    if (!dir) {
        cerr << "❌ 無法開啟根資料夾：" << root_folder << endl;
        return all_features;
    }

    bool found_subdirs = false;

    while ((ent = readdir(dir)) != nullptr) {
        string subname = ent->d_name;
        if (subname == "." || subname == "..") continue;

        string subfolder = root_folder + "/" + subname;
        if (!isDirectory(subfolder)) continue;

        if (subname.rfind("i_", 0) != 0 && subname.rfind("v_", 0) != 0)
            continue;

        found_subdirs = true;
        cout << "📂 掃描資料夾：" << subfolder << endl;

        DIR* subdir = opendir(subfolder.c_str());
        struct dirent* subent;

        while ((subent = readdir(subdir)) != nullptr) {
            string fname = subent->d_name;
            if (!isImageFile(fname, allowed_exts)) continue;

            string filepath = subfolder + "/" + fname;
            Mat img = imread(filepath, IMREAD_GRAYSCALE);
            if (img.empty()) continue;

            vector<KeyPoint> keypoints;
            Mat descriptors;
            orb->detectAndCompute(img, noArray(), keypoints, descriptors);
            if (descriptors.empty()) continue;
            
            vector<Mat> desc_vec;
            for (int i = 0; i < descriptors.rows; ++i) {
                desc_vec.push_back(descriptors.row(i));
            }

            all_features.push_back(desc_vec);
        }
        closedir(subdir);
    }

    closedir(dir);

    if (!found_subdirs) {
        cout << "📁 掃描圖片於：" << root_folder << endl;
        DIR* flat_dir = opendir(root_folder.c_str());
        if (flat_dir) {
            while ((ent = readdir(flat_dir)) != nullptr) {
                string fname = ent->d_name;
                if (!isImageFile(fname, allowed_exts)) continue;

                string filepath = root_folder + "/" + fname;
                Mat img = imread(filepath, IMREAD_GRAYSCALE);
                if (img.empty()) continue;

                vector<KeyPoint> keypoints;
                Mat descriptors;
                orb->detectAndCompute(img, noArray(), keypoints, descriptors);
                if (descriptors.empty()) continue;

                vector<Mat> desc_vec;
                for (int i = 0; i < descriptors.rows; ++i) {
                    desc_vec.push_back(descriptors.row(i));
                }

                all_features.push_back(desc_vec);
            }
            closedir(flat_dir);
        }
    }

    cout << "✅ 成功處理圖片數量: " << all_features.size() << endl;
    return all_features;
}

void trainAndSaveVocabulary(const string& folder, const string& basename, const vector<string>& exts) 
{
    vector<vector<Mat>> features = loadFeatures(folder, exts);
    if (features.empty()) {
        cerr << "⚠️ Warning: No features found in " << folder << endl;
        return;
    }

    struct Config {
        int k;
        int L;
        string suffix;
    };

    vector<Config> configs = {
        {10, 6, ""}
        //10, 4, "k10_L4"},
        //{10, 5, "k10_L5"}
    };

    for (const auto& cfg : configs) {
        cout << "🧠 Training with k = " << cfg.k << ", L = " << cfg.L << endl;
        auto start = chrono::high_resolution_clock::now();

        OrbVocabulary voc(cfg.k, cfg.L, DBoW2::TF_IDF, DBoW2::L1_NORM);
        voc.create(features);

        string out_file = basename + "_" + cfg.suffix + ".yml.gz";
        voc.save(out_file);

        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::seconds>(end - start);

        cout << "✅ Vocabulary saved: " << out_file
             << " (" << features.size() << " images, "
             << duration.count() << " seconds)" << endl << endl;
    }
}



int main() 
{
    vector<string> allowed_exts = {".jpg", ".jpeg", ".png", ".ppm", ".bmp"};

    // Hpatches
    //trainAndSaveVocabulary("/home/lab605/dataset/hpatches-sequences-release/v_london", "voc_v_london", allowed_exts);
    //trainAndSaveVocabulary("/home/lab605/dataset/hpatches-sequences-release/i_bridger", "voc_i_bridger", allowed_exts);
    //trainAndSaveVocabulary("/home/lab605/dataset/hpatches-sequences-release", "voc_all", allowed_exts);
    
    // Visual Genome
    //trainAndSaveVocabulary("/home/lab605/dataset/Visual Genome/images/VG_100K", "Genome", allowed_exts);


    // CoCo
    trainAndSaveVocabulary("/home/lab605/dataset/coco/train2017", "CoCo", allowed_exts);
    
    return 0;
}

