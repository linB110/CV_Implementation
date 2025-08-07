#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <cstring>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "DBoW2/TemplatedVocabulary.h"
#include "DBoW2/FORB.h" // ORB feature type

#include <cnpy.h>  // read npy

using namespace std;
namespace fs = std::filesystem;

#define MAX_TRAIN_NUM 30000               // Max pseudo-images
#define DESCRIPTORS_PER_IMAGE 1000        // DBoW2 requirement

// --------------------------
// Load .npy file into cv::Mat
// --------------------------
cv::Mat loadNpyAsMat(const std::string& file_path) {
    cnpy::NpyArray arr = cnpy::npy_load(file_path);
    if (arr.shape.size() != 2 || arr.shape[1] != 32 || arr.word_size != 1) {
        throw std::runtime_error("Invalid .npy format in: " + file_path);
    }

    int rows = arr.shape[0];
    cv::Mat desc(rows, 32, CV_8U);
    std::memcpy(desc.data, arr.data<unsigned char>(), rows * 32);
    return desc;
}

// --------------------------
// Load all descriptors (limit total count)
// --------------------------
vector<cv::Mat> loadORBDescriptors(const string& descriptor_folder, size_t max_total_descriptors) {
    vector<cv::Mat> all_descriptors;
    size_t total = 0;

    for (const auto& entry : fs::recursive_directory_iterator(descriptor_folder)) {
        if (entry.path().extension() == ".npy" &&
            entry.path().filename().string().find("descriptor") != std::string::npos) {

            try {
                cv::Mat npy_desc = loadNpyAsMat(entry.path().string());
                if (npy_desc.cols != 32 || npy_desc.type() != CV_8U) {
                    cerr << "⚠️ Invalid descriptor format: " << entry.path() << endl;
                    continue;
                }

                for (int i = 0; i < npy_desc.rows && total < max_total_descriptors; ++i) {
                    all_descriptors.push_back(npy_desc.row(i).clone());
                    total++;
                }

                if (total >= max_total_descriptors) break;

            } catch (const std::exception& e) {
                cerr << "❌ Failed to load: " << entry.path() << " - " << e.what() << endl;
            }
        }
    }

    cout << "✅ Loaded " << all_descriptors.size() << " descriptors from: " << descriptor_folder << endl;
    return all_descriptors;
}

// ORB vocabulary type
typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB> ORBVocabulary;

// --------------------------
// Main
// --------------------------
int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <descriptor_folder> <output_txt>" << endl;
        return -1;
    }

    string descriptor_folder = argv[1];
    string output_txt = argv[2];

    // Load ORB descriptors
    vector<cv::Mat> descriptors = loadORBDescriptors(descriptor_folder, MAX_TRAIN_NUM * DESCRIPTORS_PER_IMAGE);

    if (descriptors.empty()) {
        cerr << "❌ No valid descriptors found!" << endl;
        return -1;
    }

    // Shuffle to reduce bias
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(descriptors.begin(), descriptors.end(), g);

    // Group into pseudo-images
    vector<vector<cv::Mat>> training_data;

    for (size_t i = 0; i + DESCRIPTORS_PER_IMAGE <= descriptors.size(); i += DESCRIPTORS_PER_IMAGE) {
        vector<cv::Mat> image_desc(descriptors.begin() + i, descriptors.begin() + i + DESCRIPTORS_PER_IMAGE);
        training_data.push_back(image_desc);
        if (training_data.size() >= MAX_TRAIN_NUM) break;
    }

    cout << "📊 Training on " << training_data.size() << " pseudo-images." << endl;

    // Create and train vocabulary
    ORBVocabulary voc;
    voc.create(training_data, 10, 6, DBoW2::TF_IDF, DBoW2::L1_NORM); // same as ORB-SLAM2 default

    cout << "📦 Vocabulary created. Size: " << voc.size() << " words." << endl;

    // Save to text file (compatible with ORB-SLAM2)
    voc.saveToTextFile(output_txt);
    cout << "💾 Vocabulary saved to: " << output_txt << endl;

    return 0;
}

